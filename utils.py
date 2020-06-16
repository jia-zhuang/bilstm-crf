import os
import collections
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils


InputFeature = collections.namedtuple('InputFeature', 'input_ids, label_ids')


def read_data(fname):
    examples = []
    with open(fname) as f:
        example = collections.defaultdict(list)
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                assert len(parts) == 2
                example['token'].append(parts[0])
                example['tag'].append(parts[1])
            else:  # sentence split by empty line
                if example['token']:
                    examples.append(example)
                example = collections.defaultdict(list)
    return examples


def convert_example_to_feature(example, label_map, w2v):
    unk_id = len(w2v) - 1
    input_ids = [w2v.stoi.get(t, unk_id) for t in example['token']]
    label_ids = [label_map[l] for l in example['tag']]
    return InputFeature(
        input_ids=input_ids,
        label_ids=label_ids,
    )


class NerDataset(Dataset):
    def __init__(self, data_dir, label_list, w2v, mode='train'):
        super().__init__()
        examples = read_data(os.path.join(data_dir, f'{mode}.txt'))
        label_map = {l: i for i, l in enumerate(label_list)}

        self.features = []
        for example in examples:
            self.features.append(
                convert_example_to_feature(example, label_map, w2v)
            )
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate_fn(batch, ignore_index=0):
    batch = sorted(batch, key=lambda x: len(x.input_ids), reverse=True)
    input_ids = [torch.tensor(f.input_ids, dtype=torch.long) for f in batch]
    label_ids = [torch.tensor(f.label_ids, dtype=torch.long) for f in batch]
    seq_lengths = torch.tensor([len(f.input_ids) for f in batch], dtype=torch.long)

    input_ids = rnn_utils.pad_sequence(input_ids, padding_value=0)
    label_ids = rnn_utils.pad_sequence(label_ids, padding_value=ignore_index)

    return {
        'input_ids': input_ids,
        'label_ids': label_ids,
        'seq_lengths': seq_lengths,
    }


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


