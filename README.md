# ProLong

[[Paper](https://arxiv.org/pdf/2410.02660)] [[HF Page](https://huggingface.co/collections/princeton-nlp/prolong-66c72d55d2051a86ac7bd7e4)]

This is the homepage for **ProLong** (<u>Pr</u>incet<u>o</u>n <u>long</u>-context language models). 

ProLong is a family of long-context models that are continued trained and supervised fine-tuned from Llama-3-8B, with a maximum context window of 512K tokens. Our [main ProLong model](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct) is one of the best-performing long-context models at the 10B scale (evaluated by [HELMET](https://github.com/princeton-nlp/helmet)).

To train this strong long-context model, we conduct thorough ablations on the long-context pre-training data, SFT data, and numerous other design choices. We demonstrate our findings in our paper, [How to Train Long-Context Language Models (Effectively)](https://arxiv.org/pdf/2410.02660).

Authors: [Tianyu Gao](https://gaotianyu.xyz/about)\*, [Alexander Wettig](https://www.cs.princeton.edu/~awettig/)\*, [Howard Yen](https://howard-yen.github.io/), [Danqi Chen](https://www.cs.princeton.edu/~danqic/) (* equal contribution)

## Release Progress


- [x] ProLong models
- [x] ProLong data
- [x] Pre-training and SFT code
- [x] Sequence parallelism

## Model card

Here are some quick facts about our main ProLong model: [princeton-nlp/Llama-3-8B-ProLong-512k-Instruct](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct).
* Base model: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
* Long-context continued training: 20B tokens on 64K training data, and 20B tokens on 512K training data
* Supervised fine-tuning (SFT): [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
* Maximum context window: 512K tokens


<p align="center">
  <img width="80%" alt="image" src="https://github.com/user-attachments/assets/c31c9671-49fe-4776-91d2-de70ffd9f9a1">
</p>

<p align="center">
<em>ProLong performance on <a href="https://github.com/princeton-nlp/helmet">HELMET</a> averaged over 32K, 64K, and 128K lengths. All models are instruct models.</em>
</p>


## Download the models and packed data

All ProLong models are available on Hugging Face. All the models are based on Llama-3-8B, so any code that supports Llama-3-8B is also compatible with ProLong models.

| Model | HF Link |
|-------|---------|
| ProLong-64k-Base | [princeton-nlp/Llama-3-8B-ProLong-64k-Base](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-64k-Base) |
| ProLong-64k-Instruct | [princeton-nlp/Llama-3-8B-ProLong-64k-Instruct](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-64k-Instruct) |
| ProLong-512k-Base | [princeton-nlp/Llama-3-8B-ProLong-512k-Base](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Base) |
| ⭐ ProLong-512k-Instruct | [princeton-nlp/Llama-3-8B-ProLong-512k-Instruct](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct)  |

Our training data (packed and sampled version) are also available on Hugging Face (in [mosaicml-streaming](https://docs.mosaicml.com/projects/streaming/en/stable/index.html) format).

| Data | HF Link |
|------|---------|
| Stage 1: 64K training data (40B tokens) | [princeton-nlp/prolong-data-64K](https://huggingface.co/datasets/princeton-nlp/prolong-data-64K) |
| Stage 2: 512K training data (40B tokens)| [princeton-nlp/prolong-data-512K](https://huggingface.co/datasets/princeton-nlp/prolong-data-512K) |
| SFT: UltraChat (1B tokens) | [princeton-nlp/prolong-ultrachat-64K](https://huggingface.co/datasets/princeton-nlp/prolong-ultrachat-64K) |




## Download and prepare raw data

If you want to experiment with different data lengths or data mixtures,
We also provide the (unpacked, unfiltered, but tokenized) raw data from each domain below. 
Due to the large size of the raw data, we store it on AWS S3. To download the data, you need to have an AWS account (with an access key and a secret key). **Note that data downloading will incur a charge on your AWS account**. According to [this S3 document](https://aws.amazon.com/s3/pricing/), each GB of data downloaded incurs $0.09 and the first 100GB is free. You can download the data using the following commands:

```bash
# Install AWS CLI if you haven't already
pip install awscli

# Configure AWS CLI with your credentials (you will need an access key and a secret key from your AWS account)
aws configure

# Download the raw code repo data (concatenated by repo names from the stack v1) 
aws s3 sync s3://princeton-prolong/data_before_packing/code_repos/ /target/path/ --request-payer requester
```

Below is the available unpacked raw data (tokenized with the Llama-3 tokenizer). All data is in the [mosaicml-streaming](https://docs.mosaicml.com/projects/streaming/en/stable/index.html) format, with three fields: `domain` (`str`), `input_ids` (`int32 numpy array`, the Llama-3 tokenized document with no BOS/EOS), and `length` (`int32`, number of tokens).

| Data | Size | S3 path |
|------|------|---------|
| Code repos | 689 GB | s3://princeton-prolong/data_before_packing/code_repos/ |
| Books (SlimPajama)| 180 GB| s3://princeton-prolong/data_before_packing/books/ |
| FineWeb (sampled) | 864 GB | s3://princeton-prolong/data_before_packing/fineweb-2023-50/ |
| FineWeb-edu (sampled) | 365 GB | s3://princeton-prolong/data_before_packing/fineweb-edu-100B/ |
| OpenWebMath | 48 GB| s3://princeton-prolong/data_before_packing/openwebmath/ |
| Wikipedia (Dolma) | 14 GB | s3://princeton-prolong/data_before_packing/wikipedia/ |
| Textbooks | 1 GB | s3://princeton-prolong/data_before_packing/textbooks/ |
| Tulu-v2 | 1 GB | s3://princeton-prolong/data_before_packing/tuluv2/ |
| StackExchange (SlimPajama) | 135 GB | s3://princeton-prolong/data_before_packing/stackexchange/ |
| ArXiv (SlimPajama) | 210 GB | s3://princeton-prolong/data_before_packing/arxiv/ |


<details>
<summary>A quick guide of mosaicml-streaming</summary>

Full documentation and installation guide can be found [here](https://docs.mosaicml.com/projects/streaming/en/stable/index.html).

<pre>
<code class="language-python">>>> from streaming import LocalDataset
>>> dataset = LocalDataset("path/to/dataset")
>>> len(dataset) # number of samples
>>> dataset[0] # allow random access, use like a dictionary/JSON
{'domain': 'book', 'input_ids': array([ 1038, 19017,  2041, ...,   271, 12488,   220], dtype=uint32), 'length': 111200}</code>
</pre>

</details>



### How to filter and pack data

We use our own [datatools](https://github.com/CodeCreator/datatools) (created by Alex and Tianyu) to filter (by lengths) and pack data. `datatools` is a versatile repo that supports tokenization/packing/filtering from various raw formats (json, jsonl, hugging face, mosaicml-streaming, etc) and outputs the data in the mosaicml-streaming format.

Example usage: 
```bash
pack <input path> <output path> --pack_length <pack_length> --min_length <discard docs with less tokens> -w <workers>

# For example, pack our raw code data to 64K with 40 workers
pack data/code_repo data/code_repo-packto64k-minlen64k  --pack_length 65536 --min_length 65536 -w 40

# Our script is also compatible with distributed workflows on SLURM. The example belows uses 20 SLURM array jobs, each using 40 workers
pack data/code_repo data/code_repo-packto64k-minlen64k  --pack_length 65536 --min_length 65536 -w 40 --num_jobs 20 --slurm_array

# If you want to tokenize some raw data with text strings into tokenized data (which can then be packed). The example belows uses 20 SLURM array jobs, each using 40 workers.
# The input directory should also be of mosaic-streaming format. Each item should have a "text" field as raw strings of documents.
# You should first run at a smaller scale and check if the result looks correct
tokenize data/code_repo_text data/code_repo -w 40 --num_jobs 20 --slurm_array --tokenizer {HF tokenizer name / llama2 / llama3}

```

## How to train ProLong

<p align="center">
  <img width="80%" alt="image" src="https://github.com/user-attachments/assets/a36a7d0f-4480-4a29-80f3-208477707fb7">
</p>
<p align="center">
<em>ProLong training recipe.</em>
</p>


Our training code is built on top of Hugging Face's [Transformers](https://github.com/huggingface/transformers). Compared to the original codebase, we make the following changes:

* Support `mosaicml-streaming` formats for datasets (much faster and IO friendly).
* Support FlashAttention-2's variable-length attention (for efficient document masking). We implemented an in-batch length-sorting dataloader that balances data loads on different devices and improves training throughput.
* Support sequence parallelism (inspired by DeepSpeed Ulysses).
* Support SFT (masking out instructions) and token-averaged losses (instead of torch's standard sequence-and-device-averaged losses).
* We implemented a memory-efficient cross entropy  that allows 64K-token training of Llama-3-8B without using sequence parallelism.
* Various improvements on checkpoint resuming and logging.

#### File structures

All our code is under `training`:
* `dataset.py`: datasets and packing strategies for mosaicml-streaming data.
* `distributed_attention.py`: sequence parallelism implementation.
* `modeling_flash_llama.py`: our modified FlashAttention-2 Llama code, with support for variable-length attention, sequence parallelism, memory-efficient cross entropy, and token-averaged losses.
* `trainer.py`: our trainer derived from Hugging Face's `Trainer` with various improvements.
* `train_language_model.py`: the main training script.

#### Preparation

1. Download all the data to `datasets/`
```bash
git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-64K datasets/long-context-65536
git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-512K datasets/long-context-524288
git clone https://huggingface.co/datasets/princeton-nlp/prolong-ultrachat-64K datasets/prolong-ultrachat-64K
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

#### Training

We provide the scripts for 64K training (`train_64K.sh`), 512K training (`train_512K.sh`), and the final SFT training (`train_sft.sh`). The scripts require at least 8 GPUs (each with at least 80GB memory) to run. To run it on a local machine, simply do `bash {script_name}.sh`. If you are using SLURM in a cluster environment, you can submit the job by `sbatch {script_name}.sh`. To submit a resume-from-checkpoint job, the same script will work too.

The 512K training will load the 64K checkpoint and the optimizer state. To allow this, **please do the following**
```bash
cd {the HF checkpoint folder of the 64K model}
mv trainer_state.json trainer_state.json.backup # Otherwise the model will reload the old LR scheduler
ln -s checkpoint-5000/optimizer.pt . # Link the optimizer state so that it can be loaded; replace checkpoint-5000 to whichever that is the last checkpoint
```


#### Customization

You can read the comments in the scripts to see what customized training arguments we used. 
Here is a brief explanation of them (we skip all that are already defined in Hugging Face):
* `--cuda_empty_cache`: empty CUDA cache after each step to avoid OOM.
* `--config_overrides`: override the default HF config with specified arguments, e.g., `--config_overrides "rope_theta=8000000"`.
* `--seq_parallel_size`: sequence parallelism size. For example, `--seq_parallel_size 8` means we use 8 GPUs to handle one long sequence.
* `--apply_instruct_masks`: read the `mask` field from the dataset and mask out those tokens during instruction tuning (e.g., the instructions).
* `--token_scaled_loss`: average losses over valid training tokens instead of devices. This should be turned on during instruction tuning.

There are more options regarding FSDP, gradient checkpointing, etc. Please refer to the scripts for more details.

## Contact

Please email Tianyu (`tianyug@princeton.edu`) or Alex (`awettig@princeton.edu`) if you have any questions. If you encounter any issues with the code, models, or data, please open an issue on GitHub.


## Citation

```bibtex
@article{gao2024prolong,
  title={How to Train Long-Context Language Models (Effectively)},
  author={Gao, Tianyu and Wettig, Alexander and Yen, Howard and Chen, Danqi},
  journal={arXiv preprint arXiv:2410.02660},
  year={2024}
}
```
