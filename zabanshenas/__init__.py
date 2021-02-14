import json
import os
from zabanshenas.model import TransformerLangDetection


def get_portable_filename(filename):
    path, _ = os.path.split(__file__)
    filename = os.path.join(path, filename)
    return filename


# BASE_DIR = os.getcwd()
MODELS = {
    "v1": get_portable_filename("models/v1-415154705/")
}


def beshnas(
    text: str,
    show_topk: bool = False,
    topk: int = 5,
    progressbar: bool = False,
    model_version: str = "v1",
    model_args: dict = {}
):
    model_dir = MODELS[model_version]

    model_dir = model_dir[:-1] if model_dir.endswith("/") else model_dir
    with open(f"{model_dir}/labels.json", "r", encoding="utf-8") as fp:
        label_map = json.load(fp)

    tokenizer = TransformerLangDetection.load_tokenizer(f"{model_dir}/tokenizer.json")
    model = TransformerLangDetection.load(model_dir=model_dir, **model_args)
    top1, topk = model.predict(
        [text],
        tokenizer=tokenizer,
        topk=topk,
        progressbar=progressbar,
        label_map=label_map
    )

    if show_topk:
        probs = topk[0][0].tolist()
        indices = topk[1][0].tolist()
        labels = [label_map[idx] for idx in indices]

        result = {}
        for i, (prob, idx, label) in enumerate(zip(probs, indices, labels)):
            result[i] = {
                "prob": prob,
                "idx": idx,
                "code": label["code"],
                "name": label["name"],
            }

        return result

    prob, idx, label = top1[0][0], top1[1][0], top1[2][0]
    result = {
        "prob": prob,
        "idx": idx,
        "code": label["code"],
        "name": label["name"],
    }

    return result
