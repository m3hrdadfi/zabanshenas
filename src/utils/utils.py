import torch
import subprocess


def num_lines_in_file(fpath):
    return int(subprocess.check_output('wc -l %s' % fpath, shell=True).strip().split()[0])


def make_src_mask(
    input_ids: torch.Tensor,
    pad_idx: int = 0
):
    # input_ids shape: (batch_size, seq_len)
    # mask shape: (batch_size, 1, 1, seq_len)

    batch_size = input_ids.shape[0]

    src_mask = (input_ids != pad_idx).view(batch_size, 1, 1, -1)

    return src_mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
