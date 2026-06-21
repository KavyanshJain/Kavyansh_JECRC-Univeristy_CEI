# Week 5 — Text Generation using RNN, LSTM, and GRU

This notebook is part of my weekly assignments for the Celebal Technologies Data Science Training Program. The goal for this week was to build and compare sequence models that can learn from a text corpus and generate new text.

## What this covers

- Tokenization and n-gram sequence preparation
- Training three architectures — Vanilla RNN, LSTM, and GRU — on the same corpus
- Comparing training loss curves across 200 epochs
- Generating text from a shared seed phrase using all three models
- Temperature sampling to improve output diversity and reduce repetition

## Dataset

[Text Generator dataset, Next Word Predictor, LLMs](https://www.kaggle.com/datasets/ashishpandey2062/next-word-predictor-text-generator-dataset) it contains a large text file with plain text of various field and suitable for our RNN models training.

## Models

| Model | Key Characteristic |
|---|---|
| Vanilla RNN | Simple but struggles with long-range context |
| LSTM | Handles long-term dependencies using three gates |
| GRU | Lighter alternative to LSTM with similar performance |

## How to run
Open the notebook in Kaggle, import the dataset using [Kaggle link](https://www.kaggle.com/datasets/ashishpandey2062/next-word-predictor-text-generator-dataset) or directly upload the .txt from the repo and change the import line as per your conditions.
Run all cells top to bottom.

## Temperature Sampling

One thing I noticed was that using argmax for word prediction caused the models to repeat the same tokens after a certain point. To fix this, I added temperature sampling which adjusts the probability distribution before picking the next word. Tested at three values — 0.5, 1.0, and 1.5 — and 1.0 gave the most readable output without being too repetitive or too random.

## Results

All three models converge reasonably well on the corpus. LSTM and GRU outperform vanilla RNN in terms of final loss and text coherence. Temperature sampling at 1.0 produces the most balanced output across all three architectures.
