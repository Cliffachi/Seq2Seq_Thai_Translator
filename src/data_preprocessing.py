
import re
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# -------------------------
# Special tokens
# -------------------------
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

SPECIAL_TOKENS = {
    "PAD": PAD_token,
    "SOS": SOS_token,
    "EOS": EOS_token,
    "UNK": UNK_token,
}


# -------------------------
# Vocabulary
# -------------------------
class Language:
    def __init__(self, name):
        self.name = name

        self.word2index = {}
        self.word2count = {}

        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
            UNK_token: "UNK",
        }

        self.n_words = 4

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)


# -------------------------
# Text preprocessing
# -------------------------
def normalize_english(text):
    text = text.strip().lower()

    # Separate common punctuation into tokens
    text = re.sub(r"([.!?,;:])", r" \1", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def normalize_thai(text):
    # Preserve Thai characters and combining marks.
    # Assumes Thai has already been segmented by PyThaiNLP.
    return " ".join(text.strip().split())

def load_pairs(path, max_length=50):
    pairs = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            columns = line.split("\t")

            if len(columns) != 2:
                raise ValueError(
                    f"Line {line_number} must contain "
                    "exactly two tab-separated columns."
                )

            english, thai = columns

            english = normalize_english(english)
            thai = normalize_thai(thai)

            if not english or not thai:
                continue

            # Filter by TOKEN count, not character count
            if (
                len(english.split()) >= max_length
                or len(thai.split()) >= max_length
            ):
                continue

            pairs.append((english, thai))

    if not pairs:
        raise ValueError(
            "No valid sentence pairs found. Check the dataset path and format."
        )
    return pairs


# -------------------------
# Split dataset
# -------------------------
def split_pairs(pairs, seed=42):
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


# -------------------------
# Build vocabulary
# -------------------------
def build_vocab(pairs):
    input_lang = Language("eng")
    output_lang = Language("tha")

    # Training pairs ONLY
    for english, thai in pairs:
        input_lang.add_sentence(english)
        output_lang.add_sentence(thai)

    return input_lang, output_lang


# -------------------------
# Convert sentences to IDs
# -------------------------
def sentence_to_ids(sentence, lang):
    return [
        lang.word2index.get(word, UNK_token)
        for word in sentence.split()
    ]


def pair_to_tensors(pair, input_lang, output_lang):
    english, thai = pair

    # Encoder input: English words + EOS
    src_ids = sentence_to_ids(english, input_lang)
    src_ids.append(EOS_token)

    # Decoder sequence: SOS + Thai words + EOS
    tgt_ids = [SOS_token]
    tgt_ids.extend(sentence_to_ids(thai, output_lang))
    tgt_ids.append(EOS_token)

    return (
        torch.tensor(src_ids, dtype=torch.long),
        torch.tensor(tgt_ids, dtype=torch.long),
    )


# -------------------------
# Batch collation
# -------------------------
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_lengths = torch.tensor(
        [len(seq) for seq in src_batch],
        dtype=torch.long,
    )

    tgt_lengths = torch.tensor(
        [len(seq) for seq in tgt_batch],
        dtype=torch.long,
    )

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD_token,
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=PAD_token,
    )

    return src_batch, src_lengths, tgt_batch, tgt_lengths

# -------------------------
# Prepare everything
# -------------------------
def prepare_data(
    path="data/eng-tha.txt",
    batch_size=64,
    max_length=50,
    seed=42,
):
    pairs = load_pairs(path, max_length=max_length)

    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, seed=seed
    )

    input_lang, output_lang = build_vocab(train_pairs)

    train_data = [
        pair_to_tensors(p, input_lang, output_lang)
        for p in train_pairs
    ]

    val_data = [
        pair_to_tensors(p, input_lang, output_lang)
        for p in val_pairs
    ]

    test_data = [
        pair_to_tensors(p, input_lang, output_lang)
        for p in test_pairs
    ]

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"English vocabulary: {input_lang.n_words}")
    print(f"Thai vocabulary: {output_lang.n_words}")

    return (
        input_lang,
        output_lang,
        train_loader,
        val_loader,
        test_loader,
    )
