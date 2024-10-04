# ProLong

[[Paper]()] [[HF Page](https://huggingface.co/collections/princeton-nlp/prolong-66c72d55d2051a86ac7bd7e4)]

This is the homepage for **ProLong** (<u>Pr</u>incet<u>o</u>n <u>long</u>-context language models). 

ProLong is a family of long-context models that are continued trained and supervised fine-tuned from Llama-3-8B, with a maximum context window of 512K tokens. Our [main ProLong model](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct) is one of the best-performing long-context models at the 10B scale (evaluated by [HELMET](https://github.com/princeton-nlp/helmet) at 128K).

To train this strong long-context model, we conduct thorough ablations on the long-context pre-training data, SFT data, and numerous other design choices. We demonstrate our findings in our paper, [How to Train Long-Context Language Models (Effectively)]().


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

## Download the models and data

All ProLong models are available on Hugging Face.

| Model | HF Link |
|-------|---------|
| ProLong-64k-Base | [princeton-nlp/Llama-3-8B-ProLong-64k-Base](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-64k-Base) |
| ProLong-64k-Instruct | [princeton-nlp/Llama-3-8B-ProLong-64k-Instruct](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-64k-Instruct) |
| ProLong-512k-Base | [princeton-nlp/Llama-3-8B-ProLong-512k-Base](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Base) |
| ⭐ ProLong-512k-Instruct | [princeton-nlp/Llama-3-8B-ProLong-512k-Instruct](https://huggingface.co/princeton-nlp/Llama-3-8B-ProLong-512k-Instruct)  |

Our training data are also available on Hugging Face.

| Data | HF Link |
|------|---------|
| Stage 1: 64K training data | [princeton-nlp/prolong-data-64K](https://huggingface.co/datasets/princeton-nlp/prolong-data-64K) |
| Stage 2: 512K training data | [princeton-nlp/prolong-data-512K](https://huggingface.co/datasets/princeton-nlp/prolong-data-512K) |




## Citation

```bibtex
@article{gao2024prolong,
    title={Enabling Large Language Models to Generate Text with Citations},
    author={Gao, Tianyu and Wettig, Alexander and Yen, Howard and Chen, Danqi},
    year={2024},
}
```
