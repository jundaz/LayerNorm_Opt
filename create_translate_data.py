import torch
import log_linear_model as ll
from torch.utils import data


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def create_dateset(in_file, tar_file, num_lines, min_freq, num_steps, device, batch_size):
    input_vocab = ll.create_vocab(in_file, num_lines=num_lines, min_freq=min_freq)
    target_vocab = ll.create_vocab(tar_file, num_lines=num_lines, min_freq=min_freq)
    in_IO = open(in_file, 'r', encoding='UTF-8')
    tar_IO = open(tar_file, 'r', encoding='UTF-8')
    in_line = in_IO.readline()
    in_split = in_line.strip().split()
    tar_line = tar_IO.readline()
    tar_split = tar_line.strip().split()
    in_seq = []
    tar_seq = []
    i = 0
    while in_line and tar_line and i < num_lines:
        in_seq.append(in_split)
        tar_seq.append(tar_split)
        in_line = in_IO.readline()
        in_split = in_line.strip().split()
        tar_line = tar_IO.readline()
        tar_split = tar_line.strip().split()
        i += 1
    in_array, in_val_len = build_array_nmt(in_seq, input_vocab, num_steps)
    tar_array, tar_val_len = build_array_nmt(tar_seq, target_vocab, num_steps)
    return load_array((in_array, in_val_len, tar_array, tar_val_len), batch_size), input_vocab, target_vocab

