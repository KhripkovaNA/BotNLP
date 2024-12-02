import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from app.text_search.service import load_tfidf_model_and_index, search_texts, get_relevant_texts

# Тестовые данные
sample_raw_texts = [
    "Python - это отличный язык программирования.",
    "FastAPI позволяет создавать быстрые веб-приложения.",
    "Машинное обучение важно для современного мира.",
]
sample_processed_texts = [
    'python отличный язык программирование',
    'fastapi позволять создавать быстрый веб приложение',
    'машинный обучение важный современный мир'
]


class TestTextSearchService(unittest.TestCase):
    def setUp(self):
        """Создание тестовой среды"""
        self.temp_dir = TemporaryDirectory()
        self.mock_tfidf_folder = Path(self.temp_dir.name) / "mock_folder"
        self.mock_empty_folder = Path(self.temp_dir.name) / "mock_empty_folder"
        self.mock_tfidf_folder.mkdir(parents=True, exist_ok=True)
        self.mock_empty_folder.mkdir(parents=True, exist_ok=True)
        self.mock_model_path = self.mock_tfidf_folder / "tfidf_model.pkl"
        self.mock_matrix_path = self.mock_tfidf_folder / "tfidf_matrix.pkl"
        self.mock_texts_path = self.mock_tfidf_folder / "texts.pkl"

        # Создаем и обучаем TfidfVectorizer и генерируем TF-IDF матрицу
        vectorizer = TfidfVectorizer()
        # Обучаем модель на обработанных текстах
        self.sample_tfidf_matrix = vectorizer.fit_transform(sample_processed_texts).toarray()
        self.sample_vectorizer = vectorizer

        # Мок-сериализация модели и матрицы TF-IDF
        with open(self.mock_model_path, "wb") as f:
            pickle.dump(self.sample_vectorizer, f)
        with open(self.mock_matrix_path, "wb") as f:
            pickle.dump(self.sample_tfidf_matrix, f)
        with open(self.mock_texts_path, "wb") as f:
            pickle.dump(sample_raw_texts, f)

    def tearDown(self):
        """Очистка тестовой среды"""
        self.temp_dir.cleanup()

    def test_tfidf_matrix(self):
        """Тест: проверка корректности создания TF-IDF матрицы"""
        self.assertEqual(self.sample_tfidf_matrix.shape[0], len(sample_processed_texts))
        self.assertGreater(self.sample_tfidf_matrix.shape[1], 0)

    def test_load_tfidf_model_and_index(self):
        """Тест загрузки модели TF-IDF и матрицы"""
        vectorizer, matrix = load_tfidf_model_and_index(self.mock_model_path, self.mock_matrix_path)
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertIsInstance(matrix, np.ndarray)
        self.assertTrue(np.array_equal(matrix, self.sample_tfidf_matrix))

    def test_search_texts_valid_query(self):
        """Тест обработки корректного запроса функцией поиска"""
        query = "язык программирования"
        results = search_texts(query, self.sample_vectorizer, self.sample_tfidf_matrix, sample_raw_texts)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(r[0], str) and isinstance(r[1], float) for r in results))
        self.assertTrue("Python" in results[0][0])  # Ожидаем, что релевантный текст первый

    def test_search_texts_invalid_query(self):
        """Тест обработки некорректного запроса функцией поиска"""
        with self.assertRaises(ValueError):
            search_texts("", None, None, None)

    def test_get_relevant_texts(self):
        """Тест получения релевантных текстов"""
        query = "веб-приложения"
        results = get_relevant_texts(query, self.mock_tfidf_folder)

        # Проверяем, что возвращаются корректные данные
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

    def test_get_relevant_texts_irrelevant_query(self):
        """Тест получения релевантных текстов с нерелевантным запросом"""
        query = "абракадабра"
        results = get_relevant_texts(query, self.mock_tfidf_folder)
        self.assertTrue(all(r[1] == 0.0 for r in results))

    def test_get_relevant_texts_missing_files(self):
        """Тест обработки ошибки при отсутствии файлов"""
        with self.assertRaises(FileNotFoundError) as context:
            get_relevant_texts("веб-приложения", self.mock_empty_folder)
        self.assertIn("Следующие файлы не найдены", str(context.exception))


if __name__ == "__main__":
    unittest.main()
