# English–Thai Sequence-to-Sequence Translator

A PyTorch neural machine translation project that learns to translate English sentences into Thai. It uses a GRU encoder-decoder model with multiplicative attention and supports training, validation, checkpointing, and dataset preparation.

## Model

The sequence-to-sequence model consists of:

- Separate source and target embeddings
- A GRU encoder
- A GRU decoder
- Scaled dot-product attention between decoder states and encoder outputs
- A linear output layer that produces a target vocabulary distribution
- Padding-aware loss calculation and gradient clipping during training

## Project Structure

```text
se2seq_translator/
├── data/
│   ├── eng-tha.txt       # Tab-separated English and Thai sentence pairs
│   └── clean_txt_.py     # Utility for cleaning two-column dataset files
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Data Format

The training file must contain one sentence pair per line, with the source and target separated by a tab:

```text
Hello	สวัสดี
How are you?	คุณเป็นอย่างไรบ้าง
```

The default dataset is `data/eng-tha.txt`. Sentences are normalized and filtered before the data is split into training, validation, and test sets.

## Setup

Python 3.10 or later is recommended.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Training

Run the training script from the project root:

```powershell
python src/train.py
```

The script:

1. Loads and preprocesses `data/eng-tha.txt`.
2. Builds source and target vocabularies from the training split.
3. Creates train, validation, and test data loaders.
4. Initializes the `Seq2Seq` model.
5. Trains for the configured number of epochs.
6. Saves the best validation checkpoint to `checkpoints/seq2seq_best.pt`.

Training settings can be changed near the top of `src/train.py`:

```python
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
GRAD_CLIP = 1.0
```

The trainer automatically uses CUDA when available and otherwise uses the CPU.

## Dataset Cleaning

The utility in `data/clean_txt_.py` accepts the name of a source text file and keeps the first two tab-separated columns:

```powershell
python data/clean_txt_.py
```

For example, entering `tha` reads `data/tha.txt` and writes `data/eng-tha.txt`.

## Dependencies

- PyTorch
- NumPy

## License

Add a license for the project if it will be distributed publicly.
