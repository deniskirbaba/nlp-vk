from typing import List

import numpy as np
from minhash import MinHash


class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        self.num_permutations = num_permutations
        self.num_buckets = num_buckets
        self.threshold = threshold

    def get_buckets(self, minhash: np.ndarray) -> List[np.ndarray]:
        """
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        """
        step = max(1, round(len(minhash) / self.num_buckets))
        buckets = np.array_split(minhash, indices_or_sections=np.arange(step, len(minhash), step))
        return buckets

    def get_similar_candidates(self, buckets) -> list[tuple]:
        """
        Находит потенциально похожих кандидатов.
        Кандидаты похожи, если полностью совпадают мин хеши хотя бы в одном из бакетов.
        Возвращает список из таплов индексов похожих документов.
        """
        similar_candidates = set()
        for b_i, bucket in enumerate(buckets):
            bucket_sim = self.get_similar_matrix(bucket)
            np.fill_diagonal(bucket_sim, False)
            sim_ids = np.argwhere(bucket_sim == True)
            similar_candidates |= {(i, j) for i, j in sim_ids}
        
        return list(similar_candidates)


    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        buckets = self.get_buckets(minhash)

        similar_candidates = self.get_similar_candidates(buckets)

        return similar_candidates
