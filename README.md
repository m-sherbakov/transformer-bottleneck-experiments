# Transformer Bottleneck Experiment

How much can we compress encoder representations before a Transformer struggles to learn?

This project explores the effect of a low-dimensional bottleneck between the encoder and decoder of a classic Transformer.

## Experiment

I implemented an Encoder–Decoder Transformer in PyTorch:

- **Model dimension:** 256
- **Layers:** 1 encoder + 1 decoder
- **Attention heads:** 8
- **Task:** Reverse sequences of 10 integers (1–50)
- **Bottleneck dimensions:** 2, 4, 8, 16, 32, 64, 128, 256
- **Training:** 500 steps, Adam optimizer

A linear bottleneck compresses each encoder output vector from 256 to `n` dimensions before passing it to the decoder's Cross-Attention.

## Results

- **n = 2:** Failed to learn the task sufficiently within 500 steps.
- **n = 4:** Learned significantly more slowly.
- **n ≥ 8:** Achieved nearly perfect token accuracy.

Surprisingly, freezing the randomly initialized bottleneck produced results similar to those of a trainable bottleneck.

This suggests that the encoder and decoder can adapt to a fixed low-dimensional communication channel.

## Limitations

These are preliminary results from a single random seed on a simple task. Evaluation uses teacher forcing rather than autoregressive generation.

## Code

See [`Transformer_Experiment.ipynb`](Transformer_Experiment.ipynb) for the implementation, training experiments, and visualizations.
