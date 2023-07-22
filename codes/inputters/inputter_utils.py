# coding=utf-8
import gzip
import json
import os
import math
import random
import pickle
from functools import partial
from torch.utils.data import DataLoader, Sampler


def _norm(s):
    return ' '.join(s.strip().split())


class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class BucketingDataLoader(object):
    def __init__(self, toker, feature_dataset, batch_size,
                 bucket=100, shuffle=True, **kwargs):
        assert 'inputter_name' in kwargs
        assert 'config_name' in kwargs
        inputter_name = kwargs.pop('inputter_name')
        config_name = kwargs.pop('config_name')
        print(f'./DATA/{inputter_name}.{config_name}_persona_attention_final/data.pkl')
        # with open(f'./DATA/{inputter_name}.{config_name}_new_split_usr_sys/data.pkl', 'rb') as f:
        # with open(f'./DATA/{inputter_name}.{config_name}_new_split/data.pkl', 'rb') as f:
        # with open(f'./DATA/{inputter_name}.{config_name}_no_persona/data.pkl', 'rb') as f: 复现
        with open(f'./DATA/{inputter_name}.{config_name}_persona_attention_final/data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.toker = toker
        self.feature_dataset = feature_dataset
        self.batch_size = batch_size
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle

    def __iter__(self):
        trunc_chunk = []
        lens = []
        for feat in self.data:
            trunc_chunk.append(feat)
            lens.append(feat.input_len)

        dataset = self.feature_dataset(trunc_chunk)
        sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0,  # can test multi-worker
                            collate_fn=partial(self.feature_dataset.collate, toker=self.toker))
        yield from loader

    def __len__(self):
        return len(self.data)


class DistributedBucketingDataLoader(BucketingDataLoader):
    """ distributed version """
    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.num_replica = num_replica
        self.data = self.data[self.rank::self.num_replica]


# copy from torch.nn.utils.rnn
def my_pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0.0):
    # type: (List[Tensor], bool, float) -> Tensor

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
