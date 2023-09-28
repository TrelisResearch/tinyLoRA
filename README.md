# TinyLoRA
Allowing LLMs to be trained on more data before perplexity plateaus.

## Background
### Llama 2 models of different size
- [Llama 2 models](https://github.com/jzhang38/TinyLlama/blob/main/.github/llama2-training.png) appear not to saturate within 3 Trillion tokens.
- [TinyLlama](https://github.com/jzhang38/TinyLlama) has not saturated yet at around 1T tokens.
- [UltraTinyLlama](https://wandb.ai/metaskepsis/llama2.c/reports/loss-val-23-09-28-10-48-36---Vmlldzo1NTMzNjIz?accessToken=s2bzcye9e08jrdge6iymu1ycc99lt9tr2tzuhyco4apvg8s898c0bzhou1bfnars) with 100M parameters has saturated at 200B tokens.

The question is, can training be extended without increasing the model size (at least not significantly)?

### LoRA
[Low Rank Adapters](https://arxiv.org/abs/2106.09685) have been found to outperform full fine tuning - even, or perhaps especially, when the rank of the adapters is much smaller than the rank of the base matrices.

This suggests that data can be more efficiently embedded into language models.

However, just adding one LoRA to a base model is unlikely to extend knowledge by all that much. [although probably worth testing, and doing LoRA on all possible modules].

### Fast Feed Forward (FFF) Networks and Mixture of Experts
Mixture of Expert models allow for faster inference by (organically) segmenting the knowledge into different experts. Inference time is reduced versus a single expert (provided expert in the mixture is smaller than the single expert model).

It appears that [FFF networks](https://github.com/pbelcak/fastfeedforward) - which set up a binary tree with leaves that each are separate models - achieve all of the inference speed up of Mixture of Experts. However, FFF networks appear just as fast to train as standard transformers (which is much quicker that Mixture of Experts).

## TinyLoRA - adding a binary tree of LoRA adapters
The idea is to take a Llama 2 model that has saturated and apply and train a binary tree of LoRA adapters.

LoRA adapters are efficient to train and small in size. By using a tree of LoRA adapters (with a simple sigmoid classifier at each branch) it may be possible to achieve a non-linear increase in effective model size.

Other points:
- All LoRAs would be loaded to VRAM, increasing VRAM slightly for inference.
- Each LoRA could be applied on the fly for the generation of each token OR input embeddings could be averaged and the same LoRA used for an entire inference.
- The routers (each simply p in [0,1]) are trained along with the LoRAs.
- For single device inference, all batches would have to use the same LoRA for each token, which is a drawback. For multi-device inference, each LoRA could have its own device.