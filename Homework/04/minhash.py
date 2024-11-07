import re

import numpy as np
import pandas as pd


class MinHash:
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        return re.sub("( )+|(\n)+", " ", text).lower()

    def tokenize(self, text: str) -> set:
        text = self.preprocess_text(text)
        return set(text.split(" "))

    def get_occurrence_matrix(self, corpus_of_texts: list[str]) -> pd.DataFrame:
        """
        Получение матрицы вхождения токенов. Строки - это токены, столбы это id документов.
        id документа - нумерация в списке начиная с нуля
        """
        # convert str to set
        corpus_of_sets = list()
        for text in corpus_of_texts:
            corpus_of_sets.append(self.tokenize(self.preprocess_text(text)))

        # form vocab
        vocab = set()
        for t in corpus_of_sets:
            vocab |= t
        vocab = sorted(list(vocab))

        # calc mapping from token to idx
        t_to_i = {t: i for i, t in enumerate(vocab)}

        # transform tokens to ids
        corpus_ids = np.array([[t_to_i[t], doci] for doci, doc in enumerate(corpus_of_sets) for t in doc])

        # fill the emb matrix
        emb = np.zeros(shape=(len(vocab), len(corpus_of_sets)))
        emb[corpus_ids[:, 0], corpus_ids[:, 1]] = 1
        
        # create df
        df = pd.DataFrame(emb)
        df.sort_index(inplace=True)
        return df

    def is_prime(self, a):
        if a % 2 == 0:
            return a == 2
        d = 3
        while d * d <= a and a % d != 0:
            d += 2
        return d * d > a

    def get_new_index(self, x: int, permutation_index: int, prime_num_rows: int) -> int:
        """
        Получение перемешанного индекса.
        values_dict - нужен для совпадения результатов теста, а в общем случае используется рандом
        prime_num_rows - здесь важно, чтобы число было >= rows_number и было ближайшим простым числом к rows_number
        """
        values_dict = {"a": [3, 4, 5, 7, 8], "b": [3, 4, 5, 7, 8]}
        a = values_dict["a"][permutation_index]
        b = values_dict["b"][permutation_index]
        return (a * (x + 1) + b) % prime_num_rows

    def get_minhash_similarity(self, array_a: np.ndarray, array_b: np.ndarray) -> float:
        """
        Вовзращает сравнение minhash для НОВЫХ индексов. То есть: приходит 2 массива minhash:
            array_a = [1, 2, 1, 5, 3]
            array_b = [1, 3, 1, 4, 3]

            на выходе ожидаем количество совпадений/длину массива, для примера здесь:
            у нас 3 совпадения (1,1,3), ответ будет 3/5 = 0.6
        """
        
        return np.mean(array_a == array_b)

    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        """
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        """
        # mask out all unused elem
        triangle_mask = np.tri(len(min_hash_matrix), k=0, dtype=np.bool_)
        min_hash_matrix[triangle_mask] = -1
        sim_ids = np.argwhere(min_hash_matrix > self.threshold)

        return [(i, j) for i, j in sim_ids]

    def get_similar_matrix(self, min_hash_matrix) -> np.ndarray:
        """
        Находит похожих кандидатов. Отдает матрицу расстояний
        """
        sim = np.ones(shape=(min_hash_matrix.shape[1], min_hash_matrix.shape[1]))
        for i in range(min_hash_matrix.shape[1]):
            for j in range(i + 1, min_hash_matrix.shape[1]):
                sim[[i, j], [j, i]] = self.get_minhash_similarity(min_hash_matrix[:, i], min_hash_matrix[:, j])

        return sim

    def get_minhash(self, occurrence_matrix: pd.DataFrame) -> np.ndarray:
        """
        Считает и возвращает матрицу мин хешей. MinHash содержит в себе новые индексы.

        new index = (2*(index + 1) + 3) % 3

        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу
        [1, 0, 1]
        [1, 0, 1]
        [0, 1, 1]

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 0
        Doc2 : 2
        Doc3 : 0
        """
        # find prime_num_rows param for get_new_index method
        prime_num_rows = occurrence_matrix.shape[0]
        while not self.is_prime(prime_num_rows):
            prime_num_rows += 1
       
        signatures = np.empty(shape=(self.num_permutations, occurrence_matrix.shape[1]))
        for perm_idx in range(self.num_permutations):
            # perform permute
            perm_ids = np.array([self.get_new_index(i, perm_idx, prime_num_rows) for i in range(occurrence_matrix.shape[0])])
            occurrence_matrix["perm"] = perm_ids
            sorted_occurrence_matrix = occurrence_matrix.sort_values("perm").reset_index(drop=True)

            # find and save signatures
            signatures[perm_idx] = sorted_occurrence_matrix.idxmax().values[:-1]

        return signatures

    def run_minhash(self, corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        similar_matrix = self.get_similar_matrix(minhash)
        similar_pairs = self.get_similar_pairs(similar_matrix)
        return similar_pairs


class MinHashJaccard(MinHash):
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def get_jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """
        Вовзращает расстояние Жаккарда для двух сетов.
        """
        return len(set_a & set_b) / len(set_a | set_b)

    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        """
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        """
        # mask out all unused elem
        triangle_mask = np.tri(len(min_hash_matrix), k=0, dtype=np.bool_)
        min_hash_matrix[triangle_mask] = -1
        sim_ids = np.argwhere(min_hash_matrix > self.threshold)

        return [(i, j) for i, j in sim_ids]

    def get_similar_matrix(self, min_hash_matrix) -> np.ndarray:
        """
        Находит похожих кандидатов. Отдает матрицу расстояний
        """
        sim = np.ones(shape=(min_hash_matrix.shape[1], min_hash_matrix.shape[1]))
        for i in range(min_hash_matrix.shape[1]):
            for j in range(i + 1, min_hash_matrix.shape[1]):
                sim[[i, j], [j, i]] = self.get_jaccard_similarity(set(min_hash_matrix[:, i]), set(min_hash_matrix[:, j]))

        return sim

    def get_minhash_jaccard(self, occurrence_matrix: pd.DataFrame) -> np.ndarray:
        """
        Считает и возвращает матрицу мин хешей. Но в качестве мин хеша выписываем минимальный исходный индекс, не новый.
        В такой ситуации можно будет пользоваться расстояние Жаккрада.

        new index = (2*(index +1) + 3) % 3

        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу
        [1, 0, 1] index: 2
        [1, 0, 1] index: 1
        [0, 1, 1] index: 0

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 2
        Doc2 : 0
        Doc3 : 2

        """
        signatures = np.empty(shape=(self.num_permutations, occurrence_matrix.shape[1]))
        for perm_idx in range(self.num_permutations):
            # perform permute
            # of course, it's better to calculate permutations using hasm func over indices
            occurrence_matrix_shuffled = occurrence_matrix.sample(frac=1)
            
            # find and save signatures
            signatures[perm_idx] = occurrence_matrix_shuffled.idxmax(axis=0).values

        return signatures

    def run_minhash(self, corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash_jaccard(occurrence_matrix)
        similar_matrix = self.get_similar_matrix(minhash)
        similar_pairs = self.get_similar_pairs(similar_matrix)
        
        return similar_pairs
