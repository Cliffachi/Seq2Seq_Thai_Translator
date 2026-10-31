from pathlib import Path
import random
import os
from collections import Counter
import math

import torch
import torch.nn as nn

from data_preprocessing import (
    prepare_data,
    PAD_token,
    SOS_token,
    EOS_token,
    UNK_token,
)
from model import Seq2Seq
from visualization import (
    plot_attention_heatmap,
    plot_training_history,
)

CHECKPOINT_PATH = (
    Path(os.getenv("SEQ2SEQ_CHECKPOINT_DIR", "checkpoints"))
    / "seq2seq_best.pt"
)
# -------------------------
# Hyperparameters
# -------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
GRAD_CLIP = 1.0

EMBED_DIM = 128
HIDDEN_DIM = 256
MAX_LENGTH = 50
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Training function
# -------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for src, src_lengths, tgt, _ in loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        tgt = tgt.to(device)

        # Teacher forcing:
        # Decoder input: SOS, word1, word2, ...
        # Targets:       word1, word2, EOS, ...

        decoder_input = tgt[:, :-1]
        targets = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        # logits: [B, T-1, target_vocab_size]
        logits, _ = model(
            src,
            src_lengths,
            decoder_input,
        )

        # Flatten predictions and targets for CrossEntropyLoss
        predictions = logits.reshape(
            -1, logits.size(-1)
        )
        targets_flat = targets.reshape(-1)

        loss = criterion(predictions, targets_flat)

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training loss is NaN or infinite."
            )

        loss.backward()

        # Prevent exploding gradients
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP,
        )

        optimizer.step()

        # Accumulate token-weighted loss
        non_padding_tokens = (
            targets_flat != PAD_token
        ).sum().item()

        total_loss += loss.item() * non_padding_tokens
        total_tokens += non_padding_tokens

    return total_loss / max(total_tokens, 1)


# -------------------------
# Validation function
# -------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for src, src_lengths, tgt, _ in loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        tgt = tgt.to(device)

        # Same teacher-forcing alignment as training
        decoder_input = tgt[:, :-1]
        targets = tgt[:, 1:]

        logits, _ = model(
            src,
            src_lengths,
            decoder_input,
        )

        predictions = logits.reshape(
            -1, logits.size(-1)
        )
        targets_flat = targets.reshape(-1)

        loss = criterion(predictions, targets_flat)

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Validation loss is NaN or infinite."
            )

        non_padding_tokens = (
            targets_flat != PAD_token
        ).sum().item()

        total_loss += loss.item() * non_padding_tokens
        total_tokens += non_padding_tokens

    return total_loss / max(total_tokens, 1)


def bleu_score(reference, hypothesis, max_order=4):
    if not reference or not hypothesis:
        return 0.0

    precisions = []
    effective_orders = min(max_order, len(hypothesis), len(reference))
    for order in range(1, effective_orders + 1):
        if len(hypothesis) < order:
            return 0.0

        hypothesis_ngrams = Counter(
            tuple(hypothesis[index:index + order])
            for index in range(len(hypothesis) - order + 1)
        )
        reference_ngrams = Counter(
            tuple(reference[index:index + order])
            for index in range(len(reference) - order + 1)
        )

        clipped = sum(
            min(count, reference_ngrams[ngram])
            for ngram, count in hypothesis_ngrams.items()
        )
        total = sum(hypothesis_ngrams.values())
        precisions.append(clipped / total if total else 0.0)

    if any(precision == 0.0 for precision in precisions):
        return 0.0

    geometric_mean = math.exp(
        sum(math.log(precision) for precision in precisions) / effective_orders
    )
    brevity_penalty = 1.0
    if len(hypothesis) < len(reference):
        brevity_penalty = math.exp(
            1.0 - len(reference) / len(hypothesis)
        )

    return 100.0 * brevity_penalty * geometric_mean


@torch.no_grad()
def calculate_validation_bleu(model, loader, tgt_index2word, device):
    model.eval()
    scores = []

    for src, src_lengths, tgt, _ in loader:
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        tgt = tgt.to(device)
        encoded, hidden = model.encode(src, src_lengths)

        for batch_index in range(src.size(0)):
            decoder_hidden = (
                hidden[0][:, batch_index:batch_index + 1],
                hidden[1][:, batch_index:batch_index + 1],
            ) if isinstance(hidden, tuple) else hidden[:, batch_index:batch_index + 1]
            decoder_input = torch.tensor(
                [[SOS_token]],
                dtype=torch.long,
                device=device,
            )
            source_encoded = encoded[batch_index:batch_index + 1]
            source_ids = src[batch_index]
            hypothesis_ids = []

            for _ in range(MAX_LENGTH):
                embedded = model.tgt_embedding(decoder_input)
                decoder_output, decoder_hidden = model.decoder(
                    embedded,
                    decoder_hidden,
                )
                context, _ = model.attention(
                    source_encoded,
                    decoder_output,
                    source_ids.unsqueeze(0),
                )
                combined = torch.cat([decoder_output, context], dim=-1)
                logits = model.decoder_fc(combined)
                next_token = logits[:, -1].argmax(dim=-1)
                token_id = next_token.item()
                if token_id == EOS_token:
                    break
                if token_id not in (PAD_token, SOS_token, UNK_token):
                    hypothesis_ids.append(token_id)
                decoder_input = next_token.unsqueeze(1)

            reference_ids = [
                token_id
                for token_id in tgt[batch_index].tolist()
                if token_id not in (PAD_token, SOS_token, EOS_token, UNK_token)
            ]
            hypothesis_tokens = [
                tgt_index2word.get(token_id, "UNK")
                for token_id in hypothesis_ids
            ]
            reference_tokens = [
                tgt_index2word.get(token_id, "UNK")
                for token_id in reference_ids
            ]
            scores.append(bleu_score(reference_tokens, hypothesis_tokens))

    return sum(scores) / max(len(scores), 1)


@torch.no_grad()
def save_model_attention_example(
    model,
    loader,
    src_index2word,
    tgt_index2word,
    output_dir,
    device,
):
    model.eval()
    src, src_lengths, tgt, _ = next(iter(loader))
    src = src[:1].to(device)
    src_lengths = src_lengths[:1].to(device)
    tgt = tgt[:1].to(device)

    decoder_input = tgt[:, :-1]
    logits, attention_weights = model(src, src_lengths, decoder_input)
    predicted_ids = logits.argmax(dim=-1)[0].tolist()

    source_ids = src[0].tolist()
    source_tokens = [
        src_index2word.get(token_id, "UNK")
        for token_id in source_ids
        if token_id != PAD_token
    ]
    predicted_tokens = [
        tgt_index2word.get(token_id, "UNK")
        for token_id in predicted_ids
    ]

    weights = attention_weights[0].detach().cpu()
    weights = weights[:len(predicted_tokens)]
    weights = weights[:, :len(source_tokens)]
    plot_attention_heatmap(
        weights,
        source_tokens=source_tokens,
        target_tokens=predicted_tokens,
        save_path=Path(output_dir) / "attention.png",
    )


# -------------------------
# Main training pipeline
# -------------------------
def train():
    set_seed(SEED)
    print("Device:", DEVICE)

    # Prepare English -> Thai data
    (input_lang,
    output_lang,
    train_loader,
    val_loader,
    test_loader,) = prepare_data(
                                path="data/eng-tha.txt",
                                batch_size=BATCH_SIZE,
                                max_length=MAX_LENGTH,
                                seed=SEED,
                                )

    print("English vocabulary:", input_lang.n_words)
    print("Thai vocabulary:", output_lang.n_words)

    # Initialize model
    model = Seq2Seq(
        src_vocab_size=input_lang.n_words,
        tgt_vocab_size=output_lang.n_words,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=PAD_token,
    ).to(DEVICE)

    # Ignore padded target tokens in the loss
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_token
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    checkpoint_dir = CHECKPOINT_PATH.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    bleu_scores = []

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        bleu_score_value = calculate_validation_bleu(
            model,
            val_loader,
            output_lang.index2word,
            DEVICE,
        )
        bleu_scores.append(bleu_score_value)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"BLEU: {bleu_score_value:.2f}"
        )

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,

                # Model configuration
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "padding_idx": PAD_token,
                "src_vocab_size": input_lang.n_words,
                "tgt_vocab_size": output_lang.n_words,

                # Vocabulary mappings for inference
                "src_word2index": input_lang.word2index,
                "src_index2word": input_lang.index2word,
                "tgt_word2index": output_lang.word2index,
                "tgt_index2word": output_lang.index2word,

                # Special tokens
                "special_tokens": {
                    "PAD": PAD_token,
                    "SOS": SOS_token,
                    "EOS": EOS_token,
                    "UNK": UNK_token,
                },
            }

            checkpoint_path = CHECKPOINT_PATH
            temp_path = checkpoint_dir / "seq2seq_best.tmp"

            torch.save(checkpoint, temp_path)
            temp_path.replace(checkpoint_path)
            print(
                f"  Saved best model "
                f"(validation loss: {best_val_loss:.4f})"
            )

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_PATH}"
    )
    plot_training_history(
        train_losses,
        val_losses,
        bleu_scores,
        output_dir=Path("visualizations"),
    )
    save_model_attention_example(
        model,
        val_loader,
        input_lang.index2word,
        output_lang.index2word,
        Path("visualizations"),
        DEVICE,
    )

    # Test loader is intentionally not used during training.
    # Evaluate on it after selecting the best checkpoint.

if __name__ == "__main__":
    train()
