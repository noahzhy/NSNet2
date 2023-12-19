# NSNet2

This is an Unofficial PyTorch implementation of the paper "NSNet2: Data augmentation and loss normalization for deep noise suppression" (https://arxiv.org/abs/2008.06412).

The model is based on the NSNet2 baseline model from the [DNS-Challenge](https://github.com/microsoft/DNS-Challenge).
The pipeline is based on the paper "Towards efficient models for real-time deep noise suppression" (https://arxiv.org/abs/2101.09249).

## Description

The `model/nsnet2` model structure is same as the original model.
The `model/nsnet2_ex` model is a modified version of the original model. Includes the preprocessing and postprocessing steps in the model, but excludes the FFT and IFFT processes.

In addition, the `model/tinyNSNet` model is a modified version of the original model. The model is implemented with a small number of parameters via tensorflow. Replace the original model GRU with a FastGRNN cell. The quantized tf-lite model get 0.067 ms inference time on Apple M2 chip.

## Inference Time

| Model | Platform | Inference Time |
| :---: | :---: | :---: |
| tinyNSNet | Apple M2 | 0.067 ms |

## Attribution

Pretrained model [NSNet2](https://github.com/microsoft/DNS-Challenge/tree/v4dnschallenge_ICASSP2022/NSNet2-baseline) by [Microsoft](https://github.com/microsoft) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Citation

The baseline NSNet noise suppression:

```BibTex
@misc{braun2020data,
    title={Data augmentation and loss normalization for deep noise suppression},
    author={Sebastian Braun and Ivan Tashev},
    year={2020},
    eprint={2008.06412},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
