# Toolformer

Open-source implementation of [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) by Meta AI.

## Abstract

Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q\&A system, two different search engines, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.

## How to run

### Inference
Models are available on huggingface! [toolformer_v0](https://huggingface.co/dmayhem93/toolformer_v0_epoch2)

Quick example on how to launch it below:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained(r"dmayhem93/toolformer_v0_epoch2")
model = AutoModelForCausalLM.from_pretrained(
    r"dmayhem93/toolformer_v0_epoch2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).cuda()
generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=0
) 
```

#### Model Performance
##### v0
The model is currently able to do retrieval. In a one shot setting it will pick it up without too much hand holding.
For zero shot, adding a token bias to the <TOOLFORMER_API_START>(token index 50257) will get it started.

Token bias seems to depend on the length of context, 2.5 with minimal context, 7.5 with a lot of context, seemed to be good numbers in the brief testing.

Calculation and Calendar are a WIP, you can give it a shot, but don't expect good results.

#### Tool Integration
WIP

Tool integration into sampling is a work in progress, so you will need to manually perform the tool integration.

e.g. when it outputs <TOOLFORMER_API_START>Calculator(1 + 2)<TOOLFORMER_API_RESPONSE> you will need to input 3<TOOLFORMER_API_END> right after.

For retrieval, copy/pasting search results seems to work, but pasting results from actual retrieval is better if you have it.

To get some retrieval, here is a brief script on setting it up with some data you'll load in and retrieve from.
```python
from tools import Retriever
import json


if __name__ == '__main__':
    retriever = Retriever()
    ret_val = "location of New Orleans"
    with open('retrieval_test_data.json', encoding='utf-8') as f:
        ret_strings = json.load(f)
    print(', '.join(retriever.retrieval(
        ret_strings, ret_val, 3
    )))
```

### Data generation
Looking to make your own data?

```bash
python data_generator.py --num_devices=x, --device_id=y
```

Will let you run it without collision on x devices, so if you only have one,

```bash
python data_generator.py --num_devices=1, --device_id=0
```

Each one uses an entire GPU, so if you want to run in a node with multiple GPUs please set your CUDA_VISIBLE_DEVICES, e.g.
```bash
export CUDA_VISIBLE_DEVICES=5
python data_generator.py --num_devices=8, --device_id=5
```

The easiest way to gather multiple tools would be to make a data_generator script for each tool you want to use

finally, after you have your results, some minimal postprocessing scripts are in [this folder](data_handling)

You'll probably want to look at your data and figure out if there's any filtering needed.

For an example of what it looks like after, our first dataset generation is [here](https://huggingface.co/datasets/dmayhem93/toolformer_raw_v0), and the 
postprocessed outputs ready for HF trainer is [here](https://huggingface.co/datasets/dmayhem93/toolformer-v0-postprocessed)

## How to train

We used huggingface's run_clm.py which we put in this repository as train_gptj_toolformer.py.

We used a batch size of 32 (4/device), command used is below
```bash
deepspeed train_gptj_toolformer.py --model_name_or_path=EleutherAI/gpt-j-6B --per_device_train_batch_size=4 \
  --num_train_epochs 10 --save_strategy=epoch --output_dir=finetune_toolformer_v0 --report_to "wandb" \
  --dataset_name dmayhem93/toolformer-v0-postprocessed --tokenizer_name customToolformer \
  --block_size 2048 --gradient_accumulation_steps 1 --do_train --do_eval --evaluation_strategy=epoch \
  --logging_strategy=epoch --fp16 --overwrite_output_dir --adam_beta1=0.9 --adam_beta2=0.999 \
  --weight_decay=2e-02 --learning_rate=1e-05 --warmup_steps=100 --per_device_eval_batch_size=1 \
  --cache_dir="hf_cache" --gradient_checkpointing=True --deepspeed ds_config_gpt_j.json
```

## Citations
```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.04761,
  doi = {10.48550/ARXIV.2302.04761},
  
  url = {https://arxiv.org/abs/2302.04761},
  
  author = {Schick, Timo and Dwivedi-Yu, Jane and Dessì, Roberto and Raileanu, Roberta and Lomeli, Maria and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Toolformer: Language Models Can Teach Themselves to Use Tools},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@Article{dao2022flashattention,
    title={Flashattention: Fast and memory-efficient exact attention with io-awareness},
    author={Dao, Tri and Fu, Daniel Y and Ermon, Stefano and Rudra, Atri and R{'e}, Christopher},
    journal={arXiv preprint arXiv:2205.14135},
    year={2022}
}

@software{Liang_Long_Context_Transformer_2023,
    author = {Liang, Kaizhao},
    doi = {10.5281/zenodo.7651809},
    month = {2},
    title = {{Long Context Transformer v0.0.1}},
    url = {https://github.com/github/linguist},
    version = {0.0.1},
    year = {2023}
}
```