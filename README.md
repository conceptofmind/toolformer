# Toolformer

Open-source implementation of [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) by Meta AI.

## Abstract

Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q\&A system, two different search engines, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.

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