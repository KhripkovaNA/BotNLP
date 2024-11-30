import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from app.text_search.service import load_tfidf_model_and_index, search_texts, get_relevant_texts


class TestTextSearchService(unittest.TestCase):
    def setUp(self):
        """Создание тестовой среды"""
        self.mock_model_path = Path("mock_tfidf_model.pkl")
        self.mock_matrix_path = Path("mock_tfidf_matrix.pkl")
        self.mock_texts_path = Path("mock_texts.pkl")
        self.mock_tfidf_folder = Path("mock_folder")

        self.sample_texts = ["Тестовый текст один", "Тестовый текст два", "Тестовый текст три"]

        # Создаем и обучаем TfidfVectorizer
        vectorizer = TfidfVectorizer()
        vectorizer.fit(self.sample_texts)
        self.sample_vectorizer = vectorizer

        # Генерируем TF-IDF матрицу
        self.sample_tfidf_matrix = vectorizer.transform(self.sample_texts).toarray()

        # Мок-сериализация модели и матрицы TF-IDF
        with open(self.mock_model_path, "wb") as f:
            pickle.dump(self.sample_vectorizer, f)
        with open(self.mock_matrix_path, "wb") as f:
            pickle.dump(self.sample_tfidf_matrix, f)
        with open(self.mock_texts_path, "wb") as f:
            pickle.dump(self.sample_texts, f)

    def tearDown(self):
        """Очистка тестовой среды"""
        # Удаляем временные файлы
        if self.mock_model_path.exists():
            self.mock_model_path.unlink()
        if self.mock_matrix_path.exists():
            self.mock_matrix_path.unlink()
        if self.mock_texts_path.exists():
            self.mock_texts_path.unlink()

    def test_load_tfidf_model_and_index(self):
        """Тест загрузки модели TF-IDF и матрицы"""
        vectorizer, matrix = load_tfidf_model_and_index(self.mock_model_path, self.mock_matrix_path)
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertIsInstance(matrix, np.ndarray)
        self.assertTrue(np.array_equal(matrix, self.sample_tfidf_matrix))

    def test_search_texts_valid_query(self):
        """Тест поиска релевантных текстов"""
        query = "пример запроса"
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = np.array([[0.2, 0.3, 0.5]])
        texts = ["Тестовый текст один", "Тестовый текст два", "Тестовый текст три"]

        tfidf_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        results = search_texts(query, mock_vectorizer, tfidf_matrix, texts)

        # Проверяем, что возвращены 3 текста с релевантностью
        self.assertEqual(len(results), 3)
        for text, relevance in results:
            self.assertIsInstance(text, str)
            self.assertIsInstance(relevance, float)

    def test_search_texts_invalid_query(self):
        """Тест обработки некорректного запроса"""
        with self.assertRaises(ValueError):
            search_texts("", None, None, None)

    @patch("app.text_search.service.load_tfidf_model_and_index")
    @patch("app.text_search.service.pickle.load")
    @patch("app.text_search.service.Path.exists")
    def test_get_relevant_texts(self, mock_exists, mock_pickle_load, mock_load_tfidf):
        """Тест получения релевантных текстов"""
        mock_exists.return_value = True

        # Используем заранее подготовленные данные
        mock_load_tfidf.return_value = (self.sample_vectorizer, self.sample_tfidf_matrix)
        mock_pickle_load.return_value = self.sample_texts

        query = "пример запроса"
        results = get_relevant_texts(query)

        # Проверяем, что возвращаются корректные данные
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 0)

    @patch("app.text_search.service.Path.exists")
    def test_get_relevant_texts_missing_files(self, mock_exists):
        """Тест обработки ошибки при отсутствии файлов"""
        mock_exists.side_effect = lambda: False  # Все файлы отсутствуют
        with self.assertRaises(FileNotFoundError) as context:
            get_relevant_texts("пример запроса")
        self.assertIn("Следующие файлы не найдены", str(context.exception))


if __name__ == "__main__":
    unittest.main()
