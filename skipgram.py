import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import numpy as np
import numba as nb


class SkipGramEmbeddings(nn.Module):
    def __init__(self, n_item, n_dim, pre_trained=None):
        super(SkipGramEmbeddings, self).__init__()
        self.n_item = n_item
        self.n_dim = n_dim

        if pre_trained is None:
            self.word_embeddings = nn.Embedding(
                num_embeddings=n_item, embedding_dim=n_dim)
            self.context_embeddings = nn.Embedding(
                num_embeddings=n_item, embedding_dim=n_dim)
            self._init_weight()

        else:
            word_embeddings, context_embeddings = pre_trained
            self.word_embeddings = nn.Embedding.from_pretrained(
                word_embeddings, freeze=False)
            self.context_embeddings = nn.Embedding.from_pretrained(
                context_embeddings, freeze=False)

    def forward(self, word, context, share=False):
        word_emb = self.word_embeddings(word)

        if share:
            context_emb = self.word_embeddings(context)
        else:
            context_emb = self.context_embeddings(context)

        z = torch.bmm(
            word_emb.unsqueeze(1),
            context_emb.unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)

        return torch.sigmoid(z)

    def embedding(self, word):
        return self.word_embeddings(word)

    def embeddings(self):
        return self.word_embeddings.weight

    def get_coords(self):
        embs = [self.tag2emb, self.track2emb]
        weights = [emb.weight.detach().cpu() for emb in embs]

        return tuple(weights)

    def _init_weight(self):
        self.word_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.word_embeddings.weight.data /= self.n_dim
        self.context_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.context_embeddings.weight.data /= self.n_dim


@nb.njit(cache=True)
def _row_topk_csr(data, indices, indptr, k):
    # https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices
    m = indptr.shape[0] - 1
    max_indices = np.zeros((m, k), dtype=indices.dtype)
    max_values = np.zeros((m, k), dtype=data.dtype)

    for i in nb.prange(m):
        top_inds = np.argsort(data[indptr[i]: indptr[i + 1]])[::-1][:k]
        max_indices[i] = indices[indptr[i]: indptr[i + 1]][top_inds]
        max_values[i] = data[indptr[i]: indptr[i + 1]][top_inds]

    return max_indices, max_values


def row_topk_csr(mat, k):
    a, b = _row_topk_csr(mat.data, mat.indices, mat.indptr, k)
    return a


def sample_train_dataset(train_dataset):
    """
        Vectorized negative sampling

        train_dataset: (num_sentences, num_items)
    """
    window_size = 5

    # dataset
    train_dataset = train_dataset
    freq = np.array(np.sum(train_dataset, axis=0)).reshape(-1, )
    idx = freq > 0

    # probability
    p_item = np.zeros(freq.shape)
    p_item[idx] = freq[idx] ** 3 / 4
    p_item = p_item / np.sum(p_item)  # probability for negative sampling

    p_select = np.zeros(freq.shape)
    p_select[idx] = np.sqrt(1 / freq[idx])  # probability for sub sampling

    # sub sampling
    coo_data = train_dataset.tocoo()
    data = coo_data.col
    rows = coo_data.row

    thresholds = np.random.uniform(low=0., high=1., size=len(data))
    p_data = p_select[data]
    idx = thresholds < p_data

    data = data[idx]
    rows = rows[idx]

    # samples: row-wise random sampled non-zero columns (num_data, window_size)
    table = train_dataset[rows, :]
    table.data = np.random.rand(len(table.data))
    samples = row_topk_csr(table, window_size)

    pos_words = data.repeat(window_size)
    pos_contexts = samples.flatten()

    neg_words = data.repeat(window_size)
    neg_contexts = np.random.choice(
        np.arange(len(freq)), size=(len(data) * window_size), p=p_item)

    # remove the wrong samples (1)
    # negative samples must not appear in each row
    neg_rows = rows.repeat(window_size)
    is_appeared_in_row = train_dataset[neg_rows, neg_contexts]
    is_appeared_in_row = np.array(is_appeared_in_row, dtype=bool).reshape(-1, )

    neg_words = neg_words[~is_appeared_in_row]
    neg_contexts = neg_contexts[~is_appeared_in_row]

    # remove the wrong samples (2)
    # word and context must not be the same
    idx = pos_words == pos_contexts
    pos_words = pos_words[~idx]
    pos_contexts = pos_contexts[~idx]

    idx = neg_words == neg_contexts
    neg_words = neg_words[~idx]
    neg_contexts = neg_contexts[~idx]

    # balancing
    if (len(neg_words) > len(pos_words)):
        idx = np.random.choice(
            np.arange(len(neg_words)), size=len(pos_words))
        neg_words = neg_words[idx]
        neg_contexts = neg_contexts[idx]

    # concatenate the positive / negative samples
    words = np.concatenate([pos_words, neg_words])
    contexts = np.concatenate([pos_contexts, neg_contexts])
    labels = np.concatenate([
        np.ones((len(pos_words),), dtype=float),
        np.zeros((len(neg_words),), dtype=float)
    ])

    # create the train dataset
    sampled_train_dataset = TensorDataset(
        torch.LongTensor(words),
        torch.LongTensor(contexts),
        torch.FloatTensor(labels)
    )

    return sampled_train_dataset
