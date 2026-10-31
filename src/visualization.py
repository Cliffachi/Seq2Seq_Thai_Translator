from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _as_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu()

    if hasattr(values, "numpy"):
        return values.numpy()

    return np.asarray(values)


def _prepare_output_path(save_path):
    if save_path is None:
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finish_figure(fig, save_path=None, show=False):
    save_path = _prepare_output_path(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_training_loss(
    train_losses,
    val_losses=None,
    save_path=None,
    show=False,
):
    train_losses = _as_numpy(train_losses).reshape(-1)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Training loss")

    if val_losses is not None:
        val_losses = _as_numpy(val_losses).reshape(-1)
        if len(val_losses) != len(train_losses):
            raise ValueError(
                "train_losses and val_losses must contain the same "
                "number of epochs."
            )
        ax.plot(epochs, val_losses, marker="o", label="Validation loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show=show)


def plot_bleu_scores(
    bleu_scores,
    save_path=None,
    show=False,
):
    bleu_scores = _as_numpy(bleu_scores).reshape(-1)
    epochs = np.arange(1, len(bleu_scores) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, bleu_scores, marker="o", color="tab:green")
    ax.set_xlabel("Epoch / Evaluation")
    ax.set_ylabel("BLEU Score")
    ax.set_title("BLEU Score")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show=show)


def plot_attention_heatmap(
    attention_weights,
    source_tokens=None,
    target_tokens=None,
    save_path=None,
    show=False,
):
    weights = _as_numpy(attention_weights)
    if weights.ndim == 3:
        weights = weights[0]
    if weights.ndim != 2:
        raise ValueError(
            "attention_weights must have shape [target_length, source_length] "
            "or [1, target_length, source_length]."
        )

    target_length, source_length = weights.shape
    if source_tokens is None:
        source_tokens = [str(index) for index in range(source_length)]
    if target_tokens is None:
        target_tokens = [str(index) for index in range(target_length)]

    if len(source_tokens) != source_length:
        raise ValueError(
            "source_tokens must match the attention source dimension."
        )
    if len(target_tokens) != target_length:
        raise ValueError(
            "target_tokens must match the attention target dimension."
        )

    fig_width = max(6, source_length * 0.55)
    fig_height = max(4, target_length * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(weights, aspect="auto", cmap="viridis", vmin=0.0)

    ax.set_xticks(np.arange(source_length))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right")
    ax.set_yticks(np.arange(target_length))
    ax.set_yticklabels(target_tokens)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Target tokens")
    ax.set_title("Attention Heatmap")
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show=show)


def plot_training_history(
    train_losses,
    val_losses=None,
    bleu_scores=None,
    output_dir="visualizations",
    show=False,
):
    output_dir = Path(output_dir)
    figures = []
    figures.append(
        plot_training_loss(
            train_losses,
            val_losses,
            save_path=output_dir / "loss_curve.png",
            show=show,
        )
    )

    if bleu_scores is not None:
        figures.append(
            plot_bleu_scores(
                bleu_scores,
                save_path=output_dir / "bleu_scores.png",
                show=show,
            )
        )

    return figures


# Short aliases for callers that use metric-specific names.
plot_loss = plot_training_loss
plot_bleu = plot_bleu_scores
plot_attention = plot_attention_heatmap
