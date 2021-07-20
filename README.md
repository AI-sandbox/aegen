# ægen
Autoencoders for genomic data compression, classification, imputation, phasing and simulation.

## Overview
ægen is a meta-autoencoder which allows to customize the shape of the autoencoder and specify the desired latent space distribution. Additionally, it allows to use conditioning and/or denoising modes.

| Status 07/20 | Shape (Encoder/Decoder) | Latent Space distribution | Conditioning mode | Denoising mode |
|---|---|---|---|---|
| ✅ | Global | Gaussian | No | No |
| ✅ | Global | Multi-Bernoulli | No | No |
| ❌ | Global | Codebook (Uniform) | No | No |
| ✅ | Global | Gaussian | Yes | No |
| ✅ | Global | Multi-Bernoulli | Yes | No |
| ❌ | Global | Codebook (Uniform) | Yes | No |
| ✅ | Window-based | Gaussian | No | No |
| ✅ | Window-based | Multi-Bernoulli | No | No |
| ❌ | Window-based | Codebook (Uniform) | No | No |
| ✅ | Window-based | Gaussian | Yes | No |
| ✅ | Window-based | Multi-Bernoulli | Yes | No |
| ❌ | Window-based | Codebook (Uniform) | Yes | No |
| ✅ | Hybrid | Gaussian | No | No |
| ✅ | Hybrid | Multi-Bernoulli | No | No |
| ❌ | Hybrid | Codebook (Uniform) | No | No |
| ✅ | Hybrid | Gaussian | Yes | No |
| ✅ | Hybrid | Multi-Bernoulli | Yes | No |
| ❌ | Hybrid | Codebook (Uniform) | Yes | No |

