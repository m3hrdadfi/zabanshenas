import argparse
from pathlib import Path
import os

from tokenizers import ByteLevelBPETokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        default=None,
        metavar="dir",
        type=str,
        required=True,
        help="The directory contains the corpora."
    )
    parser.add_argument(
        "--files",
        default="**/*.txt",
        metavar="path",
        type=str,
        help="The corpora to use as training; accepts '**/*.txt' pattern."
    )
    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="The output directory, where the model will be saved."
    )
    parser.add_argument(
        "--name",
        default="bpe-bytelevel",
        type=str,
        help="The prefix name of the output model"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"The directory does not exist: {args.dir}")
        exit(1)

    paths = [str(x) for x in Path(args.dir).glob(args.files)]
    print("paths", paths)

    if not len(paths) > 0:
        print(f"The files do not exist: {args.dir}/{args.files}")
        exit(1)

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "<sep>",
        "<cls>",
        "<nl>",
        "<tab>",
        "<zwnj>",
    ]
    tokenizer.train(
        files=paths,
        vocab_size=30000,
        min_frequency=2,
        special_tokens=special_tokens + [f"[U{i}]" for i in range(1, 31)],
        show_progress=True
    )

    os.makedirs(args.out, exist_ok=True)
    tokenizer.save_model(args.out, args.name)
    tokenizer.save(f"{args.out}/tokenizer.json", pretty=True)


if __name__ == "__main__":
    main()
