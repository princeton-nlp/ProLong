import os
import torch

from streaming import StreamingDataset, Stream
import logging

from itertools import islice

from typing import Dict, Any, List, Tuple
from collections.abc import Iterator

from training.trainer import TrainingArguments

from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    single_seq: bool = field(default=False, metadata={"help": "Ignore the document boundaries and treat the whole packed sequence as a single sequence"})
    per_device_max_tokens: Optional[int] = field(default=4_294_967_296, metadata={"help": "Maximum number of tokens per device; this is to avoid some catastrophic cases where the indices or data sequences are not filtered/truncated properly in preprocessing"})
    apply_instruct_masks: bool = field(default=False, metadata={"help": "Whether to apply loss masks over the instructions (for instruction tuning). If enabled, will read the `mask` field in the data and set the corresponding labels to -100."})


class SafeStream(Stream):
    """Safe if multiple processes try to decompress the same shard."""

    def _decompress_shard_part(self, zip_info, zip_filename, raw_filename, compression):
        unique_extension = "." + str(os.getenv("SLURM_JOB_ID", "local")) + "-" + str(os.getpid())
        super()._decompress_shard_part(zip_info, zip_filename, raw_filename + unique_extension, compression)
        os.rename(raw_filename + unique_extension, raw_filename)


class DataCollator:
    def __init__(self, tokenizer, args: DataArguments):
        self.tokenizer = tokenizer
        self.args = args

    @torch.no_grad()
    def __call__(self, features):
        input_ids = []
        labels = []
        seq_lengths = []

        available_tokens = self.args.per_device_max_tokens
        for item in features:
            apply_instruct_masks = self.args.apply_instruct_masks and ("mask" in item)
            indices = item["indices"] if "indices" in item else [(0, len(item["input_ids"]))]
            if self.args.single_seq:
                indices = [(0, len(item["input_ids"]))]

            label_seq = torch.tensor(item["input_ids"], dtype=torch.long)

            for a, b in indices:
                b = a + min(b - a, available_tokens)
                if b - a > 1:
                    input_seq = torch.tensor(item["input_ids"][a:b], dtype=torch.long)
                    input_ids.append(input_seq)

                    _label = label_seq[a:b]
                    _label[0] = -100 # Don't predict the first token
                    if apply_instruct_masks:
                        # Read the `mask` field and set the corresponding labels to -100
                        mask = torch.tensor(item["mask"][a:b], dtype=torch.long)
                        _label[mask == 0] = -100
                    labels.append(_label)

                    seq_lengths.append(b - a)
                    available_tokens -= b - a
                elif available_tokens <= 0:
                    assert available_tokens == 0, "Available tokens should be non-negative"
                    break

        input_ids = torch.concat(input_ids, dim=0)
        labels = torch.concat(labels, dim=0)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)

        return dict(input_ids=input_ids,
                    attention_mask=None,
                    labels=labels,
                    seq_lengths=seq_lengths)



class SortByLengthDataset(StreamingDataset):
    def __init__(self, *args, sort_by_length_size=1, data_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sort_by_length_size = sort_by_length_size
        self.data_args = data_args

    def _negative_item_cost(self, item):
        if "indices" in item:
            return -sum(
                (end - start)**2 for start, end in item["indices"]
            )
        elif "length" in item:
            return -item["length"]**2
        else:
            return -len(item["input_ids"])**2

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.sort_by_length_size <= 1:
            yield from super().__iter__()
        else:
            iterator = super().__iter__()
            while True:
                block = list(islice(iterator, self.sort_by_length_size))
                if not block:
                    return

                yield from sorted(block, key=self._negative_item_cost)


def build_dataset(paths, training_args: TrainingArguments, data_args: DataArguments, is_training: bool) -> StreamingDataset:
    logger.info(f"Loading datasets for {'training' if is_training else 'evaluation'}")

    streams = []
    for path in paths:
        if "@" in path:
            path, proportion = path.split("@", 1)
            logger.info(f"Loading dataset from {path} with proportion {proportion}")
            streams.append(SafeStream(remote=path, local=path, proportion=float(proportion)))
        elif "#" in path:
            path, proportion = path.split("#", 1)
            logger.info(f"Loading dataset from {path} with repeat {proportion}")
            streams.append(SafeStream(remote=path, local=path, repeat=float(proportion)))
        else:
            streams.append(SafeStream(remote=path, local=path))

    epoch_size = (
        training_args.max_steps * training_args.train_batch_size * training_args.gradient_accumulation_steps *
        training_args.world_size // training_args.seq_parallel_size
    )

    num_dataloaders = max(training_args.dataloader_num_workers, 1)
    per_device_step_size = training_args.gradient_accumulation_steps * training_args.train_batch_size
    per_worker_step_size = per_device_step_size // num_dataloaders
    assert per_device_step_size % num_dataloaders == 0, "dataloader workers should divide local batch size"

    return SortByLengthDataset(
        streams=streams,
        shuffle=is_training,
        shuffle_seed=training_args.seed,
        batch_size=(training_args.train_batch_size if is_training else training_args.eval_batch_size),
        epoch_size=(epoch_size if is_training else None),
        sort_by_length_size=(per_worker_step_size if is_training else 1),
        data_args=data_args,
        replication=training_args.seq_parallel_size,
    )
