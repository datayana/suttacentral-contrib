from tokenizers import Tokenizer
from tokenizers.models import BPE

# from tokenizers import ByteLevelBPETokenizer
from tokenizers import trainers, pre_tokenizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import (
    CharDelimiterSplit,
    Punctuation,
    Whitespace,
    Sequence,
)
from tokenizers.normalizers import Lowercase

# Initialize a tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# tokenizer = ByteLevelBPETokenizer(lowercase=True)


# Train the tokenizer
bpe_train_file = "data/pali_for_bpe.txt"
# with open(bpe_train_file, "r", encoding="utf-8") as in_file:
#     bpe_chars = in_file.read()
# from collections import Counter
# print(Counter(bpe_chars))

_PUNCTUATION = [
    " ",
    "\n",
    ",",
    ".",
    ";",
    "…",
    "“",
    "”",
    "’",
    "‘",
    "?",
    "—",
    "<",
    ">",
    ":",
    "/",
    "(",
    ")",
    "–",
    "[",
    "]",
    "-",
]

tokenizer.pre_tokenizer = Sequence(
    [
        Whitespace(),
        CharDelimiterSplit("\n"),
        Punctuation(),
    ]
)  # _PUNCTUATION)

tokenizer.normalizer = Lowercase()

# Intantiate BpeTrainer
trainer = BpeTrainer(
    vocab_size=40000,
    min_frequence=2,
    show_progress=True,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

tokenizer.train(["data/pali_for_bpe.txt"], trainer)
tokenizer.save("data/tokenizer-dev.json")

with open("data/catchphrases/donotgoby.txt", "r", encoding="utf-8") as in_file:
    catchphrase = in_file.read()

    print("***")
    print(catchphrase)

    encoding = tokenizer.encode(catchphrase)
    print(encoding.tokens)
