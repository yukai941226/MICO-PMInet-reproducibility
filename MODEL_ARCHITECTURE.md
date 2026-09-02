# MICO-PMInet architecture specification

This document defines the implemented network independently of the schematic figure.

## Common input and encoder

Each animal is represented by an `8 x 467` tensor ordered as Brain, Heart, Kidney, Liver, Lung, Muscle, Spleen, and vitreous humor. A binary `8`-element mask indicates available organs. At least one organ must be retained.

| Stage | Implementation |
|---|---|
| Shared spectral encoder | Linear `467 -> 256`, LayerNorm, GELU, dropout 0.3, Linear `256 -> 256`, LayerNorm, GELU, dropout 0.3 |
| OSB | Eight organ-specific branches, each Linear `256 -> 256`, LayerNorm, GELU, dropout 0.3, Linear `256 -> 128`, LayerNorm, GELU, dropout 0.3 |
| USB | One branch with the same `256 -> 256 -> 128` structure shared across organs |
| MHA | PyTorch MultiheadAttention, embedding dimension 128, four heads, dropout 0.3, residual connection and LayerNorm |
| WMHA | Encoded organ representations are passed directly to the prediction heads |
| Organ prediction head | Linear `128 -> 64`, LayerNorm, GELU, dropout 0.3, Linear `64 -> 1`, nonnegative output clamp |
| AWA | Linear `128 -> 64`, GELU, dropout 0.3, Linear `64 -> 1`, sigmoid, mask application, then normalization across available organs |
| APA | Arithmetic mean of available organ predictions |

Linear layers use Kaiming-normal weight initialization and zero biases. LayerNorm scales are initialized to one and offsets to zero.

## Trainable parameter counts

| Architecture | Parameters |
|---|---:|
| OSB-MHA-AWA / OSB-WMHA-AWA | 1,065,346 |
| OSB-MHA-APA / OSB-WMHA-APA | 1,057,025 |
| USB-MHA-AWA / USB-WMHA-AWA | 369,154 |
| USB-MHA-APA / USB-WMHA-APA | 360,833 |

MHA modules remain present in WMHA checkpoints for state-dictionary compatibility but are skipped during the forward pass. Consequently, the stored parameter count is unchanged; effective computation differs.

## Default article optimization protocol

The default article protocol uses AdamW, learning rate 0.001, weight decay `1e-4`, batch size 32, gradient clipping at 1.0, ReduceLROnPlateau, maximum 1,000 epochs, early-stopping patience 200, main-loss weight alpha 1.0, and branch-loss beta selected from `[0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0]`. Equation 17 contains no third consistency term, so gamma is zero. Each repeated stratified split contains 48 training and 16 validation rats. See `configs/manuscript_protocol.json` and `ARTICLE_ALIGNMENT.md`.
