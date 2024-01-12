import numpy as np
import pandas as pd
import torch
import rank_bm25
import editdistance
import jieba

from rejection_sample import check_icl_set, check_icl_set_uniform

class ExampleSelector():

    def get_examples(self):
        pass

    def get_rng_state(self):
        return None

    def set_rng_state(self, state):
        pass

class StratifiedSampler(ExampleSelector):

    def __init__(self, num_shot, pool, seed=42) -> None:
        self.num_shot = num_shot
        self.pool = pool
        # self.pool = self.pool[self.pool["category"].apply(lambda x: "Source error" not in x and "Non-translation" not in x)]
        self.bins = np.linspace(-25, 0, num_shot + 1)
        self.rng = np.random.default_rng(seed)
        self.pool["bucket"] = pd.cut(self.pool["score"], bins=self.bins, include_lowest=True)
        self.buckets = self.pool.groupby("bucket")

    def get_examples(self, shuffle=True):
        examples = self.buckets.sample(n=1, random_state=self.rng)
        while not check_icl_set(examples):
            examples = self.buckets.sample(n=1, random_state=self.rng)
        if shuffle:
            examples = examples.iloc[self.rng.permutation(self.num_shot)]
        return examples

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state):
        self.rng.bit_generator.state = state

class UniformSampler(ExampleSelector):

    def __init__(self, num_shot, pool, seed=42) -> None:
        self.num_shot = num_shot
        self.pool = pool
        # self.pool = self.pool[self.pool["category"].apply(lambda x: "Source error" not in x and "Non-translation" not in x)]
        self.rng = np.random.default_rng(seed)

    def get_examples(self, shuffle=True):
        examples = self.pool.sample(n=self.num_shot, random_state=self.rng)
        while not check_icl_set_uniform(examples):
            examples = self.pool.sample(n=self.num_shot, random_state=self.rng)
        if shuffle:
            examples = examples.iloc[self.rng.permutation(self.num_shot)]
        return examples

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state):
        self.rng.bit_generator.state = state

class ExampleRetriever(ExampleSelector):

    def __init__(self, pool, example_path) -> None:
        self.pool = pool
        self.example_indices = torch.load(example_path)["example_indices"]
        self.generater = self._get_generater()

    def _get_generater(self):
        for indices in self.example_indices:
            yield self.pool.iloc[indices]

    def get_examples(self):
        return next(self.generater)

class SemanticRetriever(ExampleSelector):

    def __init__(self, num_shot, pool, src_lang, tgt_lang) -> None:
        self.num_shot = num_shot
        self.src_tokenizer, self.tgt_tokenizer = None, None
        if src_lang == "zh":
            self.src_tokenizer = lambda x: list(jieba.cut(x))
        else:
            self.src_tokenizer = lambda x: x.split(" ")
        if tgt_lang == "zh":
            self.tgt_tokenizer = lambda x: list(jieba.cut(x))
        else:
            self.tgt_tokenizer = lambda x: x.split(" ")
        self.pool = pool
        # self.pool = self.pool[self.pool["category"].apply(lambda x: "Source error" not in x and "Non-translation" not in x)]
        self.corpus = self.pool.apply(lambda x: self.src_tokenizer(x["source"]) + self.tgt_tokenizer(x["candidate"]), axis=1).to_list()
        self.bm25 = rank_bm25.BM25Okapi(self.corpus)

    def get_examples(self, src_query, translation_query):
        query = self.src_tokenizer(src_query) + self.tgt_tokenizer(translation_query)
        scores = self.bm25.get_scores(query)
        bm25_indices = np.argsort(scores)[::-1][:100]
        filterd_corpus = [self.corpus[i] for i in bm25_indices]
        edit_distance = [editdistance.eval(query, c) for c in filterd_corpus]
        ed_indices = np.argsort(edit_distance)
        indices = bm25_indices[ed_indices]

        selected_indices = []
        source_set = set()
        for idx in indices:
            if self.pool.iloc[idx]["source"] not in source_set:
                selected_indices.append(idx)
                source_set.add(self.pool.iloc[idx]["source"])
            if len(selected_indices) == self.num_shot:
                break
        selected_indices = selected_indices[::-1]
        # selected_indices = indices[:self.num_shot][::-1]
        return self.pool.iloc[selected_indices]


class StratifiedSemanticRetriever(ExampleRetriever):

    def __init__(self, num_shot, pool, src_lang, tgt_lang, seed=42) -> None:
        self.num_shot = num_shot
        self.rng = np.random.default_rng(seed)
        self.src_tokenizer, self.tgt_tokenizer = None, None
        if src_lang == "zh":
            self.src_tokenizer = lambda x: list(jieba.cut(x))
        else:
            self.src_tokenizer = lambda x: x.split(" ")
        if tgt_lang == "zh":
            self.tgt_tokenizer = lambda x: list(jieba.cut(x))
        else:
            self.tgt_tokenizer = lambda x: x.split(" ")
        self.pool = pool
        self.bins = np.linspace(-25, 0, num_shot + 1)
        self.pool["bucket"] = pd.cut(self.pool["score"], bins=self.bins, include_lowest=True)
        self.buckets = self.pool.groupby("bucket")

        self.corpus = []
        self.bm25 = []
        self.corpus_idx = []
        for _, bucket in self.buckets:
            self.corpus_idx.append(bucket.index)
            bucket_list = bucket.apply(lambda x: self.src_tokenizer(x["source"]) + self.tgt_tokenizer(x["candidate"]), axis=1).to_list()
            self.corpus.append(bucket_list)
            self.bm25.append(rank_bm25.BM25Okapi(bucket_list))

    def get_examples(self, src_query, translation_query, shuffle=True):
        query = self.src_tokenizer(src_query) + self.tgt_tokenizer(translation_query)
        selected_indices = []
        source_set = set()
        for corpus, corpus_idx, bm25 in zip(self.corpus, self.corpus_idx, self.bm25):
            scores = bm25.get_scores(query)
            bm25_indices = np.argsort(scores)[::-1][:100]
            filterd_corpus = [corpus[i] for i in bm25_indices]
            edit_distance = [editdistance.eval(query, c) for c in filterd_corpus]
            ed_indices = np.argsort(edit_distance)
            indices = bm25_indices[ed_indices]
            for idx in indices:
                if self.pool.loc[corpus_idx[idx]]["source"] not in source_set:
                    selected_indices.append(corpus_idx[idx])
                    source_set.add(self.pool.loc[corpus_idx[idx]]["source"])
                    break
        examples = self.pool.loc[selected_indices]
        if shuffle:
            examples = examples.iloc[self.rng.permutation(self.num_shot)]
        return examples


    def get_rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state):
        self.rng.bit_generator.state = state

class EditDistanceRetriever(ExampleSelector):

    def __init__(self, num_shot, pool, src_lang, tgt_lang):
        self.num_shot = num_shot
        self.src_tokenizer, self.tgt_tokenizer = None, None
        if src_lang == "zh":
            self.src_tokenizer = lambda x: list(jieba.cut(x))
        else:
            self.src_tokenizer = lambda x: x.split(" ")
        if tgt_lang == "zh":
            self.tgt_tokenizer = lambda x: list(jieba.cut(x))
        else:
            self.tgt_tokenizer = lambda x: x.split(" ")
        self.pool = pool
        self.corpus = self.pool.apply(lambda x: self.src_tokenizer(x["source"]) + self.tgt_tokenizer(x["candidate"]), axis=1).to_list()

    def get_examples(self, src_query, translation_query):
        query = self.src_tokenizer(src_query) + self.tgt_tokenizer(translation_query)
        edit_d = [editdistance.eval(query, c) for c in self.corpus]
        indices = np.argsort(edit_d)
        selected_indices = []
        source_set = set()
        for idx in indices:
            if self.pool.iloc[idx]["source"] not in source_set:
                selected_indices.append(idx)
                source_set.add(self.pool.iloc[idx]["source"])
            if len(selected_indices) == self.num_shot:
                break
        selected_indices = selected_indices[::-1]
        return self.pool.iloc[selected_indices]
