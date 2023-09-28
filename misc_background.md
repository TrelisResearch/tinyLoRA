# tinyLoRA design principles

## Background
Hmm, what kind of language model would be useful to train - something not requiring too much GPU power because I can't compete on that. Probably I can only train with a few A6000s or A100s, and actually I'd need to figure out how to parallelise properly on those.

MY LIMIT ON TRAINING:
- GPUs. I don't have the same compute...
- Best I can do is make things compute bound to crank through the data.

PROBLEMS WITH MODELS TODAY:
- The open source models are weak. Using LARGER DATASETS perhaps multi-modal, would help to do better here.
- ..

Fast Feed Forward Networks:
- Allow for faster inference (said differently, higher throughput). So you can serve a lot more people from the same GPU => CHEAPER INFERENCE.
- Seem better than MoE on all fronts because no need for noise.

LoRA:
- Allows for training at 1/3rd VRAM and 25% faster speed.

Quantization:
- Allows for training at 1/4 VRAM, unclear if it improves speed much though if things are compute bound.

TinyLlama:
- The point is to have a small model for edge devices.
- That means low VRAM.
- Ideally, that means high toks as well, which is where FFF could help.

IDEA - deep LoRA
- A key issue is packing in more data. Maybe applying LoRAs could help there? By using FFF with n LoRAs, each LoRA being the same size, you can pack in much more information because there are n LoRAs, but that doesn't drag much on inference. So the model is n times bigger, effectively.
    - Load TinyLlama.
    - Create a binary tree routing network of depth d.
    - Create a set of LoRA adapters on RAM.

A key issue is that there isn't a way to do good parallel inference when applying LoRA adapters because all batches may not be suitable... Still, it's not about speed it's really about getting more data into the model, and doing so in a way that's more efficient than just increasing the model size... It does seem, empirically that LoRA is more efficient at gathering data than a base model, unclear why.