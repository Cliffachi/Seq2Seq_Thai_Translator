
import re
from pathlib import Path

import torch

from model import Seq2Seq
from data_preprocessing import (
    PAD_token,
    SOS_token,
    EOS_token,
    UNK_token,
)


# -------------------------
# Configuration
# -------------------------
CHECKPOINT_PATH = Path("checkpoints/seq2seq_best.pt")
MAX_OUTPUT_LENGTH = 50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------
# Text preprocessing
# -------------------------
def normalize_english(text):
    # Must match training preprocessing
    text = text.strip().lower()
    text = re.sub(r"([.!?,;:])", r" \1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_to_ids(sentence, word2index):
    tokens = normalize_english(sentence).split()

    # Map unknown words to UNK
    ids = [
        word2index.get(token, UNK_token)
        for token in tokens
    ]

    # End of source sentence
    ids.append(EOS_token)

    return ids


# -------------------------
# Load model and vocabulary
# -------------------------
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run train.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = Seq2Seq(
        src_vocab_size=checkpoint["src_vocab_size"],
        tgt_vocab_size=checkpoint["tgt_vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        padding_idx=checkpoint["padding_idx"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    src_word2index = checkpoint["src_word2index"]
    tgt_index2word = checkpoint["tgt_index2word"]

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Device: {DEVICE}")

    return model, src_word2index, tgt_index2word


# -------------------------
# Translate one sentence
# -------------------------
@torch.no_grad()
def translate(
    sentence,
    model,
    src_word2index,
    tgt_index2word,
    max_length=MAX_OUTPUT_LENGTH,
):
    if not sentence.strip():
        return ""

    # Convert English sentence to IDs
    src_ids = sentence_to_ids(
        sentence,
        src_word2index,
    )

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    src_lengths = torch.tensor(
        [len(src_ids)],
        dtype=torch.long,
        device=DEVICE,
    )

    # Encode the English sentence once
    encoder_outputs, hidden = model.encode(
        src,
        src_lengths,
    )

    # Decoder begins with SOS
    decoder_input = torch.tensor(
        [[SOS_token]],
        dtype=torch.long,
        device=DEVICE,
    )

    generated_ids = []

    for _ in range(max_length):
        # Process the current decoder token
        embedded = model.tgt_embedding(decoder_input)

        decoder_outputs, hidden = model.decoder(
            embedded,
            hidden,
        )

        # Attend to the encoded English sentence
        context, _ = model.attention(
            encoder_outputs,
            decoder_outputs,
            src,
        )

        # Combine decoder output and attention context
        combined = torch.cat(
            [decoder_outputs, context],
            dim=-1,
        )

        logits = model.decoder_fc(combined)

        # Predict the next target token
        next_token = logits[:, -1, :].argmax(dim=-1)
        token_id = next_token.item()

        # Stop at EOS
        if token_id == EOS_token:
            break

        # Never display special tokens as translation words
        if token_id not in (
            PAD_token,
            SOS_token,
            UNK_token,
        ):
            generated_ids.append(token_id)

        # Feed prediction into the next decoder step
        decoder_input = next_token.unsqueeze(1)

    # Convert IDs back to Thai tokens
    output_tokens = [
        tgt_index2word.get(token_id, "UNK")
        for token_id in generated_ids
    ]

    return " ".join(output_tokens)


# -------------------------
# Interactive terminal
# -------------------------

def main():
    (
        model,
        src_word2index,
        tgt_index2word,
    ) = load_model()

    print("\nEnglish -> Thai Translator")
    print("Type 'quit' to exit.\n")

    while True:
        sentence = input("English: ").strip()

        if sentence.lower() in {"quit", "exit"}:
            print("Close program...")
            break

        if not sentence:
            continue

        translation = translate(
            sentence,
            model,
            src_word2index,
            tgt_index2word,
        )

        print(f"Thai: {translation}\n")


if __name__ == "__main__":
    main()
