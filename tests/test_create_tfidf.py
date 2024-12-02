import unittest
from io import StringIO
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from app.text_search.create_tfidf import (
    load_texts_from_folder,
    preprocess_texts,
    create_tfidf_model_and_index,
    save_tfidf_model_and_index,
)

# Тестовые данные
test_data = [
    {"text": "Python - это отличный язык программирования."},
    {"text": "FastAPI позволяет создавать быстрые веб-приложения."},
    {"text": "Машинное обучение важно для современного мира."},
]


class TestCreateTfidf(unittest.TestCase):
    def setUp(self):
        """Создание тестовой среды"""
        self.temp_dir = TemporaryDirectory()
        self.test_folder = Path(self.temp_dir.name) / "test_data"
        self.test_folder.mkdir(parents=True, exist_ok=True)
        self.test_json_file = self.test_folder / "test.json"

        # Создаём тестовый JSON
        with self.test_json_file.open("w", encoding="utf-8") as file:
            import json
            json.dump(test_data, file)

    def tearDown(self):
        """Очистка тестовой среды"""
        self.temp_dir.cleanup()

    def test_load_texts_from_folder_success(self):
        """Тест успешной загрузки текстов из JSON"""
        texts = load_texts_from_folder(self.test_folder, key="text")
        self.assertEqual(len(texts), 3)
        self.assertIn("Python - это отличный язык программирования.", texts)

    def test_load_texts_from_folder_no_folder(self):
        """Тест обработки отсутствия папки"""
        with self.assertRaises(FileNotFoundError):
            load_texts_from_folder(Path("nonexistent_folder"), key="text")

    def test_load_texts_from_folder_invalid_json(self):
        """Тест обработки некорректного JSON"""
        invalid_file = self.test_folder / "invalid.json"
        with invalid_file.open("w", encoding="utf-8") as file:
            file.write("Некорректный JSON")

        with self.assertRaises(ValueError):
            load_texts_from_folder(self.test_folder, key="text")
        invalid_file.unlink()

    def test_preprocess_texts(self):
        """Тест предобработки текстов"""
        raw_texts = ["Текст для тестирования", "Тестирование TF-IDF"]
        processed = preprocess_texts(raw_texts)
        self.assertEqual(processed, ['текст тестирование', 'тестирование tf idf'])

    def test_preprocess_texts_empty(self):
        """Тест обработки пустого списка текстов"""
        self.assertEqual(preprocess_texts([]), [])

    def test_create_tfidf_model_and_index(self):
        """Тест создания TF-IDF модели и индекса"""
        processed_texts = ['текст тестирование', 'тестирование tf idf']
        vectorizer, tfidf_matrix = create_tfidf_model_and_index(processed_texts)
        self.assertEqual(len(vectorizer.get_feature_names_out()), 4)
        self.assertEqual(tfidf_matrix.shape, (2, 4))

    def test_create_tfidf_model_and_index_empty(self):
        """Тест обработки пустого списка текстов при создании TF-IDF"""
        with self.assertRaises(ValueError):
            create_tfidf_model_and_index([])

    @patch("sys.stdout", new_callable=StringIO)
    def test_save_tfidf_model_and_index_mocked(self, mock_stdout):
        """Тест успешного создания и сохранения TF-IDF модели в тестовой директории"""

        # Вызов метода
        save_tfidf_model_and_index(self.test_folder, self.test_folder)

        # Вывод из mock_stdout
        output = mock_stdout.getvalue()

        # Убедимся, что файл действительно был сохранён
        tfidf_model_file = self.test_folder / "tfidf_model.pkl"
        self.assertTrue(tfidf_model_file.exists(), "TF-IDF модель не была сохранена в файле.")
        tfidf_matrix_file = self.test_folder / "tfidf_matrix.pkl"
        self.assertTrue(tfidf_matrix_file.exists(), "TF-IDF модель не была сохранена в файле.")

        # Проверка вывода
        self.assertIn("TF-IDF индекс успешно создан и сохранён", output)


if __name__ == "__main__":
    unittest.main()
