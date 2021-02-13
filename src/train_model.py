import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer

import numpy as np
from tqdm import tqdm

import time
import os
import shutil
import glob


from models import TransformerLangDetection
from data_loader import LangDataset
from utils.utils import (
    count_parameters,
    make_src_mask
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(data_args, model_args, training_args):
    print(f"DataTrainingArguments: \n{data_args.to_json()}")
    print(f"ModelArguments: \n{model_args.to_json()}")
    print(f"TrainingArguments: \n{training_args.to_json()}")

    print("Creating tokenizer ...")
    assert os.path.exists(data_args.tokenizer_file), "tokenizer_file does not exists"

    tokenizer = Tokenizer.from_file(data_args.tokenizer_file)
    if data_args.enable_padding:
        tokenizer.enable_padding(
            direction="right",
            pad_id=tokenizer.encode(data_args.pad_token).ids[0],
            pad_type_id=0,
            pad_token=data_args.pad_token,
            length=data_args.max_seq_len,
            pad_to_multiple_of=None
        )

        tokenizer.enable_truncation(
            data_args.max_seq_len,
            stride=0,
            strategy='longest_first'
        )

    print("Creating dataset ...")
    train_dataset = LangDataset(
        data_args.train_file,
        tokenizer,
        x_label=data_args.x_label,
        y_label=data_args.y_label,
        max_data=data_args.max_data
    )
    valid_dataset = LangDataset(
        data_args.valid_file,
        tokenizer,
        x_label=data_args.x_label,
        y_label=data_args.y_label,
        max_data=data_args.max_data
    )

    print(
        f"Train dataset sample:\n"
        f"input_texts: {train_dataset[0]['input_texts'][:100]} ...\n"
        f"  input_ids: {train_dataset[0]['input_ids'][:100]}\n"
        f"    targets: {train_dataset[0]['targets']}\n"
    )
    print(
        f"Validation dataset sample:\n"
        f"input_texts: {train_dataset[0]['input_texts'][:100]} ...\n"
        f"  input_ids: {train_dataset[0]['input_ids'][:100]}\n"
        f"    targets: {train_dataset[0]['targets']}\n"
    )

    print("Creating dataloader ...")
    train_loader = DataLoader(train_dataset, batch_size=data_args.train_batch_size, shuffle=data_args.train_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=data_args.valid_batch_size, shuffle=data_args.valid_shuffle)

    print(f"Train dataloader shape: {next(iter(train_loader))['input_ids'].shape}")
    print(f"Validation dataloader shape: {next(iter(valid_loader))['input_ids'].shape}")

    print("Creating model ...")
    model = TransformerLangDetection(
        n_vocab=len(tokenizer.get_vocab()),
        pad_idx=tokenizer.encode(data_args.pad_token).ids[0],
        d_model=model_args.d_model,
        heads=model_args.heads,
        n_position=model_args.n_position,
        ff_d_ff=model_args.ff_d_ff,
        ff_activation=model_args.ff_activation,
        ff_bias1=model_args.ff_bias1,
        ff_bias2=model_args.ff_bias2,
        n_layers=model_args.n_layers,
        n_classes=model_args.n_classes,
        dropout_rate=model_args.dropout_rate,
        scale_wte=model_args.scale_wte,
        keep_attention=model_args.keep_attention
    ).to(device)
    print()
    print(f"Model has {count_parameters(model):,} parameters")
    print()

    criterion = nn.CrossEntropyLoss()
    lr = training_args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def train(epoch, steps=0):
        model.train()

        total_loss = []
        total_acc = []

        train_loss = 0
        train_acc = 0

        start_time = time.time()

        for step, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_ids, targets = data["input_ids"], data["targets"]
            input_masks = make_src_mask(input_ids)
            batch_size = input_ids.shape[0]

            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            input_masks = input_masks.to(device)
            targets = targets.to(device)

            if model_args.keep_attention:
                outputs, _ = model(input_ids, input_masks)
            else:
                outputs = model(input_ids, input_masks)

            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            _loss = loss.item()
            _, preds = torch.max(outputs, 1)
            _acc = torch.sum(preds == targets).item() / batch_size

            total_loss.append(_loss)
            total_acc.append(_acc)

            train_loss += _loss
            train_acc += _acc

            if step > 0 and step % training_args.log_interval == 0:
                elapsed = time.time() - start_time

                train_loss = train_loss / training_args.log_interval
                train_acc = train_acc / training_args.log_interval

                print(
                    f"\tepoch {epoch:3d} | "
                    f"{step:7d}/{len(train_loader):7d} batches | "
                    f"lr {scheduler.get_last_lr()[0]:.7f} | "
                    f"ms/batch {elapsed:5.2f} | "
                    f"loss {train_loss:.4f} | "
                    f"acc {train_acc * 100:.1f} | "
                )

                train_loss = 0
                train_acc = 0
                start_time = time.time()

            steps += step

        scheduler.step()
        return total_loss, total_acc, steps

    def evaluate():
        model.eval()
        total_loss = []
        total_acc = []

        for data in tqdm(valid_loader, total=len(valid_loader)):
            input_ids, targets = data["input_ids"], data["targets"]
            input_masks = make_src_mask(input_ids)
            batch_size = input_ids.shape[0]

            input_ids = input_ids.to(device)
            input_masks = input_masks.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                if model_args.keep_attention:
                    outputs, _ = model(input_ids, input_masks)
                else:
                    outputs = model(input_ids, input_masks)
                    
                loss = criterion(outputs, targets)

                _loss = loss.item()
                _, preds = torch.max(outputs, 1)
                _acc = torch.sum(preds == targets).item() / batch_size

                total_loss.append(_loss)
                total_acc.append(_acc)

        return total_loss, total_acc

    best_loss = float("inf")
    steps = 0
    for epoch in range(1, training_args.n_epochs + 1):
        start_time = time.time()

        print()
        print("Training ...")
        train_loss, train_acc, steps = train(epoch, steps=steps)

        print()
        print("Evaluating ...")
        valid_loss, valid_acc = evaluate()

        train_loss = np.mean(train_loss)
        valid_loss = np.mean(valid_loss)

        train_acc = np.mean(train_acc)
        valid_acc = np.mean(valid_acc)

        print()
        print('-' * 90)
        print(
            f"epoch {epoch:3d} | "
            f"time {time.time() - start_time:5.2f} | "
            f"(train) loss {train_loss:.4f} | (train) acc {train_acc * 100:.1f} | "
            f"(valid) loss {valid_loss:.4f} | (valid) acc {valid_acc * 100:.1f} | "
        )
        print('-' * 90)
        print()

        if valid_loss < best_loss:
            best_loss = valid_loss

            checkpoint_dir = training_args.checkpoint_dir
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{steps}")
            os.makedirs(checkpoint_path, exist_ok=True)

            checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
            checkpoints = sorted(checkpoints, key=lambda n: int(n.split("-")[-1]))
            if len(checkpoints) >= training_args.n_checkpoints:
                shutil.rmtree(checkpoints[0])

            print(f"Saving checkpoing {checkpoint_path}")
            model.save(checkpoint_path)


if __name__ == "__main__":
    from config import DataTrainingArguments, ModelArguments, TrainingArguments

    main(DataTrainingArguments(), ModelArguments(), TrainingArguments())
