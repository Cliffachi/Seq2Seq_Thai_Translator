# Seq2Seq English-to-Thai Translator

A PyTorch sequence-to-sequence neural machine translation project that translates English text into tokenized Thai text. The model uses a unidirectional GRU encoder, a GRU decoder, and scaled dot-product attention with learned projections.

## Features

- English and Thai vocabulary creation from tab-separated training data
- GRU encoder-decoder model with attention
- Teacher-forced training with validation loss tracking
- Automatic best-checkpoint selection
- Greedy, interactive terminal translation
- Automatic CUDA selection when a compatible GPU is available
- Optional PyThaiNLP tokenization for source Thai data preparation

## Project Layout

```text
se2seq_translator/
|-- data/
|   |-- eng-tha.txt       # English<TAB>Thai training pairs
|   |-- tha.txt           # Optional raw source dataset
|   `-- clean_txt_.py     # Dataset conversion and Thai tokenization utility
|-- src/
|   |-- data_preprocessing.py
|   |-- model.py
|   |-- train.py
|   `-- inference.py
|-- checkpoints/
|   `-- seq2seq_best.pt
|-- requirements.txt
`-- README.md
```

## Dataset Format

Place each sentence pair on its own line in a tab-separated UTF-8 file:

```text
hello	สวัสดี
how are you ?	คุณ สบายดี ไหม ?
```

The expected default path is `data/eng-tha.txt`.

English text is normalized to lowercase and punctuation is separated into tokens. Thai text is split on whitespace and is therefore expected to already be word-tokenized. To convert a raw `data/tha.txt` file into a tokenized `data/eng-tha.txt` file, run:

```powershell
python data/clean_txt_.py
```

Enter `tha` when prompted. The utility keeps the first two tab-separated columns and uses PyThaiNLP's `newmm` tokenizer for Thai text.

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned dependencies are:

- PyTorch 2.15.0
- NumPy 2.5.4
- PyThaiNLP 0.13.0

## Training

From the repository root, run:

```powershell
python src/train.py
```

The training pipeline:

1. Loads and validates `data/eng-tha.txt`.
2. Randomly splits the data into 80% training, 10% validation, and 10% test sets.
3. Builds vocabularies from the training split only.
4. Trains with teacher forcing and padding-aware cross-entropy loss.
5. Clips gradients and saves the best validation checkpoint atomically.

Default hyperparameters are defined at the top of `src/train.py`:

| Setting | Default |
|---|---:|
| Batch size | 64 |
| Learning rate | 0.001 |
| Epochs | 30 |
| Gradient clipping | 1.0 |
| Embedding dimension | 128 |
| Hidden dimension | 256 |
| Maximum sentence length | 50 tokens |

### Checkpoints

By default, training saves the best model to:

```text
checkpoints/seq2seq_best.pt
```

The directory is created automatically if it does not exist. To save checkpoints elsewhere, set the `SEQ2SEQ_CHECKPOINT_DIR` environment variable before training:

```powershell
$env:SEQ2SEQ_CHECKPOINT_DIR = "C:\path\to\checkpoints"
python src/train.py
```

On subsequent runs in the same PowerShell session, clear the override to return to the project directory:

```powershell
Remove-Item Env:SEQ2SEQ_CHECKPOINT_DIR
```

Inference reads from `checkpoints/seq2seq_best.pt` by default. When a custom checkpoint directory is used, copy or move the generated `seq2seq_best.pt` into the project's `checkpoints/` directory before running inference.

## Interactive Translation

After a compatible checkpoint is available in `checkpoints/seq2seq_best.pt`, run:

```powershell
python src/inference.py
```

Example:

```text
English -> Thai Translator
Type 'quit' to exit.

English: hello
Thai: สวัสดี
```

Type `quit` or `exit` to close the translator. Translation uses greedy decoding for up to 50 output tokens and maps words outside the training vocabulary to the `UNK` token.

## Model Overview

The model consists of separate English and Thai embedding tables, a unidirectional GRU encoder, a GRU decoder, scaled dot-product attention, and a linear output projection:

```text
English tokens
      |
   Embedding
      |
      GRU encoder
      |
Encoder outputs ---- Attention ---- Decoder GRU
                              \          |
                               Context  |
                                  \     |
                         Concatenation   |
                                        |
                                Linear output
                                        |
                               Thai token logits
```

Padded sequence positions are excluded from the training loss. The vocabulary reserves IDs for padding, start-of-sequence, end-of-sequence, and unknown tokens.

## Notes

- The output quality depends heavily on the coverage and quality of `data/eng-tha.txt`.
- The checker uses only sequence length in tokens; it filters sentences with at least 50 tokens in either language.
- Test data is prepared by the pipeline, but the training script does not run a final test-set evaluation.
- Checkpoint paths should be made consistent before training and inference are run on a different machine.
