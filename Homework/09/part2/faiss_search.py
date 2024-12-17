import pickle
from typing import List, Optional

import faiss
import numpy as np
from part1.search_engine import Document, SearchResult
from sentence_transformers import SentenceTransformer


class FAISSSearcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'

    def build_index(self, documents: List[Document]) -> None:
        """
        TODO: Реализовать создание FAISS индекса

        1. Сохранить документы
        2. Получить эмбеддинги через model.encode()
        3. Нормализовать векторы (faiss.normalize_L2)
        4. Создать индекс:
            - Создать quantizer = faiss.IndexFlatIP(dimension)
            - Создать индекс = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            - Обучить индекс (train)
            - Добавить векторы (add)
        """
        self.documents = documents
        embeddings = self.model.encode(["\n".join((doc.title, doc.text)) for doc in self.documents])
        faiss.normalize_L2(embeddings)

        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 3)
        self.index.train(embeddings)
        self.index.add(embeddings)

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса

        1. Сохранить в pickle:
            - documents
            - индекс (faiss.serialize_index)
        """
        with open(path, "wb") as f:
            pickle.dump((self.documents, faiss.serialize_index(self.index)), f)

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса

        1. Загрузить из pickle:
            - documents
            - индекс (faiss.deserialize_index)
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents, self.index = data[0], faiss.deserialize_index(data[1])

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск

        1. Получить эмбеддинг запроса
        2. Нормализовать вектор
        3. Искать через index.search()
        4. Вернуть найденные документы
        """
        query_emb = self.model.encode(query)[None, :]
        faiss.normalize_L2(query_emb)

        distances, ids = self.index.search(query_emb, top_k)
        print(distances, ids)
        return [
            SearchResult(
                self.documents[doc_id].id,
                np.clip(
                    float(distances[0][i]), -1, 1
                ),  # introduce clip, bcs sometimes have values > 1
                self.documents[doc_id].title,
                self.documents[doc_id].text,
            )
            for i, doc_id in enumerate(ids[0])
        ]

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        TODO: Реализовать batch-поиск

        1. Получить эмбеддинги всех запросов
        2. Нормализовать векторы
        3. Искать через index.search()
        4. Вернуть результаты для каждого запроса
        """
        queries_emb = self.model.encode(queries)
        faiss.normalize_L2(queries_emb)

        distances, ids = self.index.search(queries_emb, top_k)

        return [
            [
                SearchResult(
                    self.documents[doc_id].id,
                    float(distances[query_idx][i]),
                    self.documents[doc_id].title,
                    self.documents[doc_id].text,
                )
                for i, doc_id in enumerate(ids[query_idx])
            ]
            for query_idx in range(len(queries))
        ]
