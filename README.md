# English-to-Hindi Machine Translation using Seq2Seq (Sutskever Architecture)

This repository contains a PyTorch implementation of the classic Sequence-to-Sequence (Seq2Seq) encoder-decoder architecture with LSTMs, as described by Sutskever et al., applied to translate sentences from **English to Hindi**.

## 📌 Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Preprocessing & Tokenization](#preprocessing--tokenization)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results & Evaluation](#results--evaluation)
- [How to Use & Translate](#how-to-use--translate)

---

## ⚡ Features
- **Custom Word-Level Tokenizer**: Built a robust custom tokenizer to handle English and Hindi vocabularies separately.
- **Optimized Sequences**: Dynamic padding, sentence length constraints (max 30 tokens), and vocabulary capping to prevent GPU Out-of-Memory (OOM) issues.
- **Source Sequence Reversal**: Reverses source English sentences (as recommended in the original Sutskever paper) to place the first words of the source sequence close to the decoder's start.
- **Mixed-Precision Training**: Utilizes PyTorch Autocast (`float16`) and GradScaler for fast training on T4/GPU runtimes.
- **Evaluation with BLEU**: Includes evaluation using Bilingual Evaluation Understudy (BLEU) scores with smoothing algorithms.

---

## 📊 Dataset
We utilize the popular **English-Hindi Dataset** containing ~130,000 sentence pairs.
- **Source**: Kaggle (`preetviradiya/english-hindi-dataset`)
- **Format**: CSV mapping English text directly to Hindi translations.

--- 

## 🛠️ Preprocessing & Tokenization
1. **Text Normalization**: Punctuation filtering, lowercasing (English), and splitting.
2. **Vocabulary Limiting**: 
   - English Vocabulary: 66,117 unique words
   - Hindi Vocabulary: Capped to the top **20,000** most frequent words to optimize output layer computation. Out-of-vocabulary words are mapped to `<unk>` (index 20001).
3. **Sequence Formatting**:
   - Hindi target sentences are wrapped with `<sos>` (Start of Sequence) and `<eos>` (End of Sequence) tokens.
   - Sentences longer than 30 tokens are filtered out to keep training fast and memory footprint low.

---

## 🏗️ Model Architecture
The network implements a Sequence-to-Sequence model using PyTorch `nn.Module`:

* **Encoder**: Multi-layer LSTM mapping source sequences to context vector (hidden and cell states).
* **Decoder**: Multi-layer LSTM using teacher forcing during training to generate target words sequentially.
* **Seq2Seq Wrapper**: Coordinates data flow between Encoder and Decoder.

### Hyperparameters
- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Number of LSTM Layers**: 2
- **Dropout**: 0.5
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross-Entropy Loss (ignoring padding tokens)

---

## 📈 Training Details
- **Epochs**: 15
- **Batch Size**: 128
- **Hardware Acceleration**: Mixed Precision (FP16) on CUDA via `torch.amp`.
- **Gradient Clipping**: Stabilized LSTM training by clipping gradients to a maximum norm of 1.0.
- **Saved Model Weights**: `sutskever_eng_to_hindi.pt`

---

## 🎯 Results & Evaluation
- **Final Loss**: ~3.0
- **Validation BLEU Score**: **2.39%** *(Tested on 500 random evaluation samples)*

### Sample Translations
| English | Model Translation (Hindi) |
| :--- | :--- |
| **Help!** | की अनुमति दी गई |
| **How are you?** | निवृत्त हैं |
| **I am going home.** | मैं तुम्हारे लिए निवृत्त |

*Note: Performance can be further improved with attention mechanisms (e.g., Bahdanau attention) or training for more epochs.*

---

## References
 
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks.* NeurIPS. https://arxiv.org/abs/1409.3215
