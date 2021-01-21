# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

from argparse import ArgumentError
import hashlib
import json
import os
import sys
from shutil import copyfile
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from pytorch_lightning.metrics.functional.classification import f1_score, auroc
from pytorch_lightning.metrics.classification import F1

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device, strict=False):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if strict:
                correct += ((p.gt(0) == y).all().float() * batch_weights.reshape((-1, 1))).sum().item()
            else:
                correct += ((p.gt(0) == y).float() * batch_weights.reshape((-1, 1))).sum().item()
            # if p.size(1) == 1:
            #     correct += (p.gt(0).eq(y).float() * batch_weights).sum().item()
            # else:
            #     correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    return correct / total


def get_ece(ps, ys):
    acc_mat = (ps.gt(.5) == ys).float()
    eces = []
    bin_weights_per_class = []
    for i in np.arange(0, 1, .1):
        if i == .9:
            bin_idx = ps.ge(i) & ps.le(i + .1)
        else:
            bin_idx = ps.ge(i) & ps.lt(i + .1)
        bin_count_per_class = bin_idx.sum(axis=0)
        per_class_bin_acc = (acc_mat * bin_idx).sum(axis=0) / bin_count_per_class
        per_class_bin_conf = (ps * bin_idx).sum(axis=0) / bin_count_per_class
        bin_weights_per_class.append(bin_count_per_class / ps.shape[0])
        eces.append(abs(per_class_bin_acc - per_class_bin_conf))
    return (torch.stack(eces, axis=0) * torch.stack(bin_weights_per_class, axis=0)).nansum(axis=0)


def get_metrics(network, loader, weights, device, name, mode='full'):
    print('Start Evaluation')
    correct = 0
    strict_correct = 0
    total = 0
    strict_total = 0
    weights_offset = 0
    ys = []
    ps = []
    sigmoid = nn.Sigmoid()

    network.eval()
    with torch.no_grad():
        t = tqdm(iter(loader), leave=False, total=len(loader))
        for i, data in enumerate(t):
            x, y = data
            if mode=='skip' and i >= 100:
                break
            x = x.to(device)
            y = y.to(device)
            p = sigmoid(network.predict(x))
            ys.append(y)
            ps.append(p)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            strict_correct += ((p.gt(.5) == y).all().float() * batch_weights.reshape((-1, 1))).sum().item()
            correct += ((p.gt(.5) == y).float() * batch_weights.reshape((-1, 1))).sum().item()
            total += p.size(0) * p.size(1)
            strict_total += batch_weights.sum().item()
        ps = torch.cat(ps).to(device)
        ys = torch.cat(ys).to(device)
        eces = get_ece(ps, ys).item()
        # micro_f1 = f1_score(ps.gt(.5).float(), ys, num_classes=None, class_reduction='micro').item()
        # macro_f1 = f1_score(ps.gt(.5).float(), ys, num_classes=None, class_reduction='macro').item()
        aucs = []
        micro_f1 = []
        macro_f1 = []
        for d in range(ps.size(1)):
            micro = F1(num_classes=2, average='micro')
            macro = F1(num_classes=2, average='macro')
            micro_f1.append(micro(ps[:, d].gt(.5).cpu().long(), ys[:, d].cpu().long()).item())
            macro_f1.append(macro(ps[:, d].gt(.5).cpu().long(), ys[:, d].cpu().long()).item())
            aucs.append(auroc(ps[:, d], ys[:, d]).item())
    network.train()
    results = {
        f'{name}_acc': correct / total,
        f'{name}_strict_acc': strict_correct / strict_total,
        f'{name}_auc': aucs,
        f'{name}_micro_f1': micro_f1,
        f'{name}_macro_f1': macro_f1,
        f'{name}_eces': eces
    }
    return results


def get_metric(network, loader, weights, device, metric):
    if weights is not None:
        raise ArgumentError("Non-uniform weights not supported")

    network.eval()
    ys = []
    ps = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            ys.extend(y)
            ps.extend(sigmoid(network.predict(x)))
        ps = torch.stack(ps).to(device)
        ys = torch.stack(ys).to(device)
        if metric == 'micro_f1':
            result = f1_score(ps, ys, num_classes=None, class_reduction='micro')
        elif metric == 'macro_f1':
            result = f1_score(ps, ys, num_classes=None, class_reduction='macro')
        elif metric == 'auroc':
            result = []
            for d in range(ps.size(1)):
                result.append(auroc(ps[:, d], ys[:, d]))
    network.train()
    return result


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
