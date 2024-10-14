# ProLong

[[Paper](https://arxiv.org/pdf/2410.02660)] [[HF Page](https://huggingface.co/collections/princeton-nlp/prolong-66c72d55d2051a86ac7bd7e4)]

This is the homepage for **ProLong** (<u>Pr</u>incet<u>o</u>n <u>long</u>-context language models). 

ProLong is a family of long-context models that are continued trained and supervised fine-tuned from Llama-3-8B, with a maximum context window of 512K tokens. Our [main ProLong model](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct) is one of the best-performing long-context models at the 10B scale (evaluated by [HELMET](https://github.com/princeton-nlp/helmet)).

To train this strong long-context model, we conduct thorough ablations on the long-context pre-training data, SFT data, and numerous other design choices. We demonstrate our findings in our paper, [How to Train Long-Context Language Models (Effectively)](https://arxiv.org/pdf/2410.02660).

Authors: [Tianyu Gao](https://gaotianyu.xyz/about)\*, [Alexander Wettig](https://www.cs.princeton.edu/~awettig/)\*, [Howard Yen](https://howard-yen.github.io/), [Danqi Chen](https://www.cs.princeton.edu/~danqic/) (* equal contribution)

## Release Progress


- [x] ProLong models
- [x] ProLong data
- [ ] Pre-training and SFT code
- [ ] Sequence parallelism

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
| Books | 180 GB| s3://princeton-prolong/data_before_packing/books/ |
| FineWeb (sampled) | 864 GB | s3://princeton-prolong/data_before_packing/fineweb-2023-50/ |
| FineWeb-edu (sampled) | 365 GB | s3://princeton-prolong/data_before_packing/fineweb-edu-100B/ |
| OpenWebMath | 48 GB| s3://princeton-prolong/data_before_packing/openwebmath/ |
| Wikipedia (Dolma) | 14 GB | s3://princeton-prolong/data_before_packing/wikipedia/ |
| Textbooks | 1 GB | s3://princeton-prolong/data_before_packing/textbooks/ |
| Tulu-v2 | 1 GB | s3://princeton-prolong/data_before_packing/tuluv2/ |

More raw data (StackExchange, Arxiv) will be released soon!

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

## How to train ProLong

<p align="center">
  <img width="80%" alt="image" src="https://github.com/user-attachments/assets/a36a7d0f-4480-4a29-80f3-208477707fb7">
</p>
<p align="center">
<em>ProLong training recipe.</em>
</p>



Coming soon!

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
