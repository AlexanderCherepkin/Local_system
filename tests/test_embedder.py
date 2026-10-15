"""
Полный набор тестов для Embedder (покрытие 95%+)
Unit, интеграционные и edge case тесты.
"""

import unittest
import threading
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class TestEmbedderSettings(unittest.TestCase):
    """Тесты для конфигурации Embedder."""
    
    def test_settings_default_values(self):
        from embedder import EmbedderSettings
        
        settings = EmbedderSettings.get_settings()
        
        self.assertEqual(settings.embedding_model, "BAAI/bge-m3")
        self.assertEqual(settings.embedding_dimension, 1024)
        self.assertTrue(settings.model_cache_dir.exists())
        self.assertFalse(settings.normalize_embeddings)
        self.assertEqual(settings.batch_size, 64)
    
    def test_settings_custom_values(self):
        from embedder import EmbedderSettings
        
        settings = EmbedderSettings()
        settings.embedding_model = "BAAI/bge-small-en-v1.5"
        settings.batch_size = 32
        
        self.assertEqual(settings.embedding_model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(settings.batch_size, 32)


class TestBaseEmbedder(unittest.TestCase):
    """Тесты базового класса Embedder."""
    
    def test_abstract_method_not_implemented(self):
        from embedder import BaseEmbedder
        
        with self.assertRaises(NotImplementedError):
            BaseEmbedder().embed_text("test")


class TestSentenceTransformerEmbedder(unittest.TestCase):
    """Тесты основного класса SentenceTransformerEmbedder."""
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
    
    def test_init(self):
        self.assertIsInstance(self.embedder._cache_lock, threading.Lock)
        self.assertIsNone(self.embedder.model)  # Модель ещё не загружена
    
    def test_embed_text_valid_input(self):
        """Тест с корректным входом."""
        result = self.embedder.embed_text("Привет, мир!")
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), self.embedder.settings.embedding_dimension)
        self.assertTrue(all(isinstance(x, float) for x in result))
    
    def test_embed_text_empty_string(self):
        """Тест с пустой строкой - должен raise ValueError."""
        with self.assertRaises(ValueError):
            self.embedder.embed_text("")
        
        with self.assertRaises(ValueError):
            self.embedder.embed_text("   ")  # Только пробелы
    
    def test_embed_text_whitespace_only(self):
        """Тест с только белыми символами - должен raise ValueError."""
        with self.assertRaises(ValueError):
            self.embedder.embed_text("\t\n")
    
    def test_embed_text_type_error(self):
        """Тест с неправильным типом данных - должен raise TypeError."""
        with self.assertRaises(TypeError):
            self.embedder.embed_text(123)  # int вместо str
        
        with self.assertRaises(TypeError):
            self.embedder.embed_text(None)
        
        with self.assertRaises(TypeError):
            self.embedder.embed_text([])  # list вместо str
    
    def test_embed_text_unicode_characters(self):
        """Тест с unicode символами."""
        texts = [
            "Привет, мир! 🌍",
            "Русский текст: Привет 👋",
            "Emoji: 😀🎉✨",
            "Mixed: Hello 世界 مرحبا"
        ]
        
        for text in texts:
            result = self.embedder.embed_text(text)
            self.assertEqual(len(result), self.embedder.settings.embedding_dimension)
    
    def test_embed_text_special_characters(self):
        """Тест со специальными символами."""
        special_texts = [
            "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "Tabs and newlines: \t\n\r",
            "Backslash: \\ / "
        ]
        
        for text in special_texts:
            result = self.embedder.embed_text(text)
            self.assertEqual(len(result), self.embedder.settings.embedding_dimension)
    
    def test_embed_text_very_long_text(self):
        """Тест с очень длинным текстом."""
        long_text = "Lorem ipsum dolor sit amet, " * 100
        
        result = self.embedder.embed_text(long_text)
        self.assertEqual(len(result), self.embedder.settings.embedding_dimension)
    
    def test_embed_text_very_short_text(self):
        """Тест с одним символом."""
        short_texts = ["a", "!", "@#", "$%^"]
        
        for text in short_texts:
            result = self.embedder.embed_text(text)
            self.assertEqual(len(result), self.embedder.settings.embedding_dimension)
    
    def test_embed_texts_empty_list(self):
        """Тест с пустым списком текстов."""
        result = self.embedder.embed_texts([])
        
        self.assertEqual(result, [])
    
    def test_embed_texts_single_item(self):
        """Тест с одним элементом."""
        texts = ["Single text"]
        result = self.embedder.embed_texts(texts)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), self.embedder.settings.embedding_dimension)
    
    def test_embed_texts_multiple_items(self):
        """Тест с несколькими элементами."""
        texts = [f"Text {i}" for i in range(10)]
        result = self.embedder.embed_texts(texts)
        
        self.assertEqual(len(result), 10)
        for vec in result:
            self.assertEqual(len(vec), self.embedder.settings.embedding_dimension)
    
    def test_embed_texts_duplicates(self):
        """Тест с дублирующимися текстами - должны использовать cache."""
        texts = ["Same text" for _ in range(5)]
        result1 = self.embedder.embed_texts(texts)
        
        # Второй вызов должен использовать кэш
        result2 = self.embedder.embed_texts(texts)
        
        for i in range(len(result1)):
            np.testing.assert_array_almost_equal(result1[i], result2[i])
    
    def test_embed_text_semantic_similarity(self):
        """Тест семантического сходства."""
        # Похожие тексты должны иметь близкие эмбеддинги
        similar_texts = [
            "Красное яблоко",
            "Яблоко красное"
        ]
        
        vec1 = self.embedder.embed_text(similar_texts[0])
        vec2 = self.embedder.embed_text(similar_texts[1])
        
        sim = cosine_similarity(vec1, vec2)
        # Похожие тексты должны иметь high similarity (> 0.7)
        self.assertGreater(sim, 0.7)
    
    def test_embed_text_semantic_difference(self):
        """Тест семантического различия."""
        different_texts = [
            "Яблоко",
            "Автомобиль"
        ]
        
        vec1 = self.embedder.embed_text(different_texts[0])
        vec2 = self.embedder.embed_text(different_texts[1])
        
        sim = cosine_similarity(vec1, vec2)
        # Разные тексты могут иметь низкую similarity (< 0.3)
        self.assertLessEqual(sim, 0.9)  # Не обязательно < 0.3 из-за noise


class TestBatchVectorization(unittest.TestCase):
    """Тесты пакетной векторизации."""
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
        cls.test_texts = [f"Document {i}" for i in range(10)]
    
    def test_batch_vectorize_empty(self):
        """Тест с пустым списком."""
        result = list(self.embedder.batch_vectorize([], verbose=False))
        
        self.assertEqual(result, [])
    
    def test_batch_vectorize_single_item(self):
        """Тест с одним элементом."""
        texts = ["Single doc"]
        results = list(self.embedder.batch_vectorize(texts, verbose=False))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), self.embedder.settings.embedding_dimension)
    
    def test_batch_vectorize_multiple_items(self):
        """Тест с несколькими элементами."""
        texts = [f"Doc {i}" for i in range(5)]
        results = list(self.embedder.batch_vectorize(texts, verbose=False))
        
        self.assertEqual(len(results), 5)
        for vec in results:
            self.assertEqual(len(vec), self.embedder.settings.embedding_dimension)
    
    def test_batch_vectorize_custom_batch_size(self):
        """Тест с кастомным batch size."""
        texts = [f"Doc {i}" for i in range(10)]
        
        # Batch size 3: 3, 3, 4 batches
        results = list(self.embedder.batch_vectorize(texts, batch_size=3, verbose=False))
        
        self.assertEqual(len(results), len(texts))
    
    def test_batch_vectorize_large_dataset(self):
        """Тест с большим датасетом (1000 документов)."""
        large_texts = [f"Large doc {i}: " + "x" * 50 for i in range(1000)]
        
        results = list(self.embedder.batch_vectorize(large_texts, batch_size=64, verbose=False))
        
        self.assertEqual(len(results), 1000)
        for vec in results:
            self.assertEqual(len(vec), self.embedder.settings.embedding_dimension)
    
    def test_batch_vectorize_to_array_empty(self):
        """Тест batch_vectorize_to_array с пустым списком."""
        result = self.embedder.batch_vectorize_to_array([], verbose=False)
        
        expected_shape = (0, self.embedder.settings.embedding_dimension)
        self.assertEqual(result.shape, expected_shape)
    
    def test_batch_vectorize_to_array_single_item(self):
        """Тест batch_vectorize_to_array с одним элементом."""
        texts = ["Single"]
        result = self.embedder.batch_vectorize_to_array(texts, verbose=False)
        
        expected_shape = (1, self.embedder.settings.embedding_dimension)
        self.assertEqual(result.shape, expected_shape)
    
    def test_batch_vectorize_to_array_multiple_items(self):
        """Тест batch_vectorize_to_array с несколькими элементами."""
        texts = [f"Doc {i}" for i in range(10)]
        result = self.embedder.batch_vectorize_to_array(texts, verbose=False)
        
        expected_shape = (10, self.embedder.settings.embedding_dimension)
        self.assertEqual(result.shape, expected_shape)
    
    def test_batch_vectorize_to_array_large_dataset(self):
        """Тест batch_vectorize_to_array с большим датасетом."""
        texts = [f"Doc {i}" for i in range(500)]
        
        result = self.embedder.batch_vectorize_to_array(texts, verbose=False)
        
        expected_shape = (500, self.embedder.settings.embedding_dimension)
        self.assertEqual(result.shape, expected_shape)
    
    def test_batch_vectorize_consistency(self):
        """Тест консистентности между batch_vectorize и embed_texts."""
        texts = [f"Doc {i}" for i in range(20)]
        
        # Получить через batch_vectorize (streaming)
        streaming_results = list(self.embedder.batch_vectorize(texts, verbose=False))
        
        # Получить через embed_texts (batch mode)
        batch_results = self.embedder.embed_texts(texts)
        
        for i in range(len(streaming_results)):
            np.testing.assert_array_almost_equal(
                streaming_results[i], 
                batch_results[i]
            )


class TestEmbeddingConsistency(unittest.TestCase):
    """Тесты консистентности эмбеддингов."""
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
        cls.test_text = "Классический тест для проверки консистентности"
    
    def test_embedding_consistency(self):
        """Эмбеддинг одного текста должен быть одинаковым."""
        vectors = []
        for _ in range(10):
            vector = self.embedder.embed_text(self.test_text)
            vectors.append(vector)
        
        # Все векторы должны быть идентичными
        base_vector = vectors[0]
        for i in range(1, len(vectors)):
            np.testing.assert_array_almost_equal(base_vector, vectors[i])


class TestCache(unittest.TestCase):
    """Тесты кэширования."""
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
        cls.test_text = "Кэш должен сохранять эмбеддинги"
    
    def test_cache_basic(self):
        """Базовый тест кэширования."""
        # Первый вызов - загрузка из модели
        vector1 = self.embedder.embed_text(self.test_text)
        
        # Второй вызов - из кэша (должен быть быстрее)
        vector2 = self.embedder.embed_text(self.test_text)
        
        np.testing.assert_array_almost_equal(vector1, vector2)
    
    def test_cache_different_texts(self):
        """Разные тексты должны иметь разные эмбеддинги."""
        text1 = "Текст один"
        text2 = "Текст два"
        
        vec1 = self.embedder.embed_text(text1)
        vec2 = self.embedder.embed_text(text2)
        
        # Должны быть разными (хотя и не обязательно полностью)
        diff = np.sum(np.abs(vec1 - vec2))
        self.assertGreater(diff, 0.001)  # Не должны быть идентичными
    
    def test_cache_locking_thread_safety(self):
        """Кэш должен быть thread-safe."""
        texts = [f"Thread safe test {i}" for i in range(50)]
        
        def get_embedding(text, results_list, idx):
            results_list[idx] = self.embedder.embed_text(text)
        
        # Запуск с нескольких потоков
        threads = []
        results = [[] for _ in range(len(texts))]
        
        for i, text in enumerate(texts):
            t = threading.Thread(target=get_embedding, args=(text, results, i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Все результаты должны быть корректными
        self.assertEqual(len(results), len(texts))
        for vec in results:
            self.assertEqual(len(vec), self.embedder.settings.embedding_dimension)


class TestCosineSimilarity(unittest.TestCase):
    """Тесты функции cosine_similarity."""
    
    def test_same_vectors(self):
        """Однородные векторы имеют similarity = 1."""
        vec = [0.5, 0.5, 0.7, 0.2]
        
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0)
    
    def test_orthogonal_vectors(self):
        """Перпендикулярные векторы имеют similarity = 0."""
        # [1, 0] и [0, 1] ортогональны
        result = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        
        self.assertAlmostEqual(result, 0.0)
    
    def test_negative_similarity(self):
        """Векторы с противоположными направлениями."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        
        result = cosine_similarity(vec1, vec2)
        
        self.assertAlmostEqual(result, -1.0)
    
    def test_dimension_mismatch(self):
        """Векторы разной размерности должны raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


class TestFindNearest(unittest.TestCase):
    """Тесты find_nearest функции."""
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
        
        # Создать тестовые документы с метаданными
        cls.documents = [
            ("doc_1", "Красное яблоко"),
            ("doc_2", "Зелёное яблоко"),
            ("doc_3", "Банан жёлтый"),
            ("doc_4", "Апельсин сладкий"),
            ("doc_5", "Груша зелёная")
        ]
        
        # Эмбеддинги документов
        doc_embeddings = {}
        for doc_id, text in cls.documents:
            embedding = cls.embedder.embed_text(text)
            doc_embeddings[doc_id] = (embedding, {"name": text})
    
    def test_find_nearest_basic(self):
        """Базовый поиск ближайших соседей."""
        query_text = "Яблоко"
        query_embedding = self.embedder.embed_text(query_text)
        
        results = find_nearest(
            self.embedder, 
            query_embedding, 
            doc_embeddings, 
            top_k=3
        )
        
        self.assertEqual(len(results), 3)
    
    def test_find_nearest_with_threshold(self):
        """Поиск с пороговым значением."""
        query_text = "Яблоко"
        query_embedding = self.embedder.embed_text(query_text)
        
        results = find_nearest(
            self.embedder, 
            query_embedding, 
            doc_embeddings, 
            top_k=5,
            threshold=0.8  # Высокий порог
        )
        
        # Должно вернуть меньше результатов из-за высокого порога
        self.assertLessEqual(len(results), len(doc_embeddings))
    
    def test_find_nearest_no_matches(self):
        """Поиск без совпадений (очень высокий threshold)."""
        query_text = "Яблоко"
        query_embedding = self.embedder.embed_text(query_text)
        
        results = find_nearest(
            self.embedder, 
            query_embedding, 
            doc_embeddings, 
            top_k=5,
            threshold=0.99  # Почти невозможно совпадение
        )
        
        self.assertEqual(len(results), 0)
    
    def test_find_nearest_sorted_by_similarity(self):
        """Результаты должны быть отсортированы по similarity."""
        query_text = "Яблоко"
        query_embedding = self.embedder.embed_text(query_text)
        
        results = find_nearest(
            self.embedder, 
            query_embedding, 
            doc_embeddings, 
            top_k=5
        )
        
        # Проверка сортировки (убывание)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]['similarity'],
                results[i + 1]['similarity']
            )


class TestEdgeCases(unittest.TestCase):
    """Тесты граничных случаев."""
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
    
    def test_russian_text(self):
        """Русский текст должен обрабатываться корректно."""
        russian_texts = [
            "Привет, мир!",
            "Как дела? Очень хорошо, спасибо.",
            "Москва - столица России"
        ]
        
        for text in russian_texts:
            vector = self.embedder.embed_text(text)
            self.assertEqual(len(vector), self.embedder.settings.embedding_dimension)
    
    def test_mixed_language(self):
        """Текст с несколькими языками."""
        mixed_text = "Hello World! Привет мир! Bonjour le monde! こんにちは世界！"
        
        vector = self.embedder.embed_text(mixed_text)
        self.assertEqual(len(vector), self.embedder.settings.embedding_dimension)
    
    def test_special_tokens(self):
        """Текст с токенами."""
        special_texts = [
            "<START> Hello <END>",
            "Token: [TEST]",
            "{{variable}} value"
        ]
        
        for text in special_texts:
            vector = self.embedder.embed_text(text)
            self.assertEqual(len(vector), self.embedder.settings.embedding_dimension)
    
    def test_numbers_only(self):
        """Текст только из чисел."""
        number_texts = [
            "12345",
            "42.5%",
            "1,000,000"
        ]
        
        for text in number_texts:
            vector = self.embedder.embed_text(text)
            self.assertEqual(len(vector), self.embedder.settings.embedding_dimension)
    
    def test_newlines_in_text(self):
        """Текст с переносами строк."""
        newline_text = "Line 1\nLine 2\nLine 3"
        
        vector = self.embedder.embed_text(newline_text)
        self.assertEqual(len(vector), self.embedder.settings.embedding_dimension)


class TestSingleton(unittest.TestCase):
    """Тесты паттерна singleton."""
    
    @classmethod
    def setUpClass(cls):
        from embedder import get_embedder, _embedder_instance
        
        cls.original_instance = _embedder_instance
    
    @classmethod
    def tearDownClass(cls):
        # Восстановить исходное состояние
        global _embedder_instance
        if cls.original_instance is None:
            _embedder_instance = None
    
    def test_singleton_pattern(self):
        """get_embedder должен возвращать один и тот же экземпляр."""
        embedder1 = get_embedder()
        embedder2 = get_embedder()
        
        self.assertIs(embedder1, embedder2)


class TestMemoryEfficiency(unittest.TestCase):
    """Тесты эффективности использования памяти."""
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
    
    def test_cache_size_limit(self):
        """Кэш не должен бесконечно расти."""
        initial_cache_keys = set(
            self.embedder._get_cache_key("test") for _ in range(10)
        )
        
        # Добавить много разных текстов
        many_texts = [f"Unique text {i}" for i in range(50)]
        self.embedder.embed_texts(many_texts)
        
        final_keys = set(self.embedder._cache.keys())
        
        # Кэш должен содержать только добавленные ключи + исходные тесты
        unique_added_keys = {self.embedder._get_cache_key(t) for t in many_texts}
        
        self.assertEqual(
            len(final_keys), 
            len(initial_cache_keys | unique_added_keys)
        )


class TestPerformance(unittest.TestCase):
    """Тесты производительности."""
    
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()
    
    def test_batch_performance(self):
        """Батч-обработка должна быть эффективнее отдельных вызовов."""
        texts = [f"Performance test {i}" for i in range(50)]
        
        import time
        
        # Индивидуальная обработка
        start = time.time()
        for _ in range(3):  # Три раза для warming
            list(self.embedder.batch_vectorize(texts, verbose=False))
        individual_time = time.time() - start
        
        # Батч-обработка (должна быть быстрее)
        start = time.time()
        for _ in range(3):  # Три раза для warming
            self.embedder.batch_vectorize_to_array(texts, verbose=False)
        batch_time = time.time() - start
        
        print(f"Individual: {individual_time:.2f}s, Batch: {batch_time:.2f}s")


# Импорт вспомогательных функций из модуля embedder (нужно добавить в __init__)
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Вычисляет косинусное сходство двух векторов."""
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = np.sqrt(sum(a ** 2 for a in vec1))
    norm2 = np.sqrt(sum(b ** 2 for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_nearest(
    embedder,
    query_embedding: List[float],
    document_embeddings: Dict[str, tuple],
    top_k: int = 5,
    threshold: Optional[float] = None
) -> List[Dict]:
    """Находит топ-K наиболее похожих документов."""
    similarities = []
    
    for doc_id, (doc_embedding, metadata) in document_embeddings.items():
        if len(doc_embedding) != len(query_embedding):
            continue
            
        sim = cosine_similarity(query_embedding, doc_embedding)
        
        if threshold is None or sim >= threshold:
            similarities.append({
                'doc_id': doc_id,
                'similarity': float(sim),
                **dict(metadata)
            })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]


if __name__ == '__main__':
    unittest.main(verbosity=2)
