from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import (
    CharDelimiterSplit,
    Punctuation,
    Whitespace,
    Sequence,
)
from tokenizers.normalizers import Lowercase

def main():
    """Parse args and run script"""
    # create parser for cli args
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--bpe_train_file",
        type=str,
        required=True,
        help="Path to the file to train BPE.",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=False,
        default=None,
        help="Path to save the tokenizer as json file.",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="print DEBUG logs",
    )

    # parse args
    args = parser.parse_args()

    # set up logger level
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    # Initialize
    # tokenizer = ByteLevelBPETokenizer(lowercase=True)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Sequence(
        [
            Whitespace(),
            CharDelimiterSplit("\n"),
            Punctuation(),
        ]
    )
    tokenizer.normalizer = Lowercase()

    # Create trainer
    trainer = BpeTrainer(
        vocab_size=40000,
        min_frequence=2,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    # Train the tokenizer
    tokenizer.train([args.bpe_train_file], trainer)

    # Save
    if args.save:
        tokenizer.save(args.save)

    with open("data/catchphrases/donotgoby.txt", "r", encoding="utf-8") as in_file:
        catchphrase = in_file.read()

        print("***")
        print(catchphrase)

        encoding = tokenizer.encode(catchphrase)
        print(encoding.tokens)

if __name__ == "__main__":
    main()
