import unittest
from io import StringIO
from unittest.mock import patch, MagicMock
from pathlib import Path
from app.text_search.create_tfidf import (
    load_texts_from_folder,
    preprocess_texts,
    create_tfidf_model_and_index,
    save_tfidf_model_and_index,
    DATA_FOLDER,
)


class TestCreateTfidf(unittest.TestCase):
    def setUp(self):
        """Создание тестовой среды"""
        self.test_folder = Path("test_data")
        self.test_folder.mkdir(exist_ok=True)
        self.test_json_file = self.test_folder / "test.json"
        self.sample_data = [{"text": "Тестовый файл"}, {"text": "Тестирование TF-IDF"}]

        # Создаём тестовый JSON
        with self.test_json_file.open("w", encoding="utf-8") as file:
            import json
            json.dump(self.sample_data, file)

    def tearDown(self):
        """Очистка тестовой среды"""
        for file in self.test_folder.iterdir():
            file.unlink()
        self.test_folder.rmdir()

    def test_load_texts_from_folder_success(self):
        """Тест успешной загрузки текстов из JSON"""
        texts = load_texts_from_folder(self.test_folder, key="text")
        self.assertEqual(len(texts), 2)
        self.assertIn("Тестовый файл", texts)

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

    @patch("app.text_search.create_tfidf.load_texts_from_folder")
    @patch("app.text_search.create_tfidf.preprocess_texts")
    @patch("app.text_search.create_tfidf.create_tfidf_model_and_index")
    @patch("app.text_search.create_tfidf.pickle.dump")
    @patch("sys.stdout", new_callable=StringIO)
    def test_save_tfidf_model_and_index(self, mock_stdout, mock_pickle_dump, mock_create_tfidf,
                                        mock_preprocess_texts, mock_load_texts):
        """Тест успешного создания и сохранения TF-IDF модели"""
        # Мок-выводы для теста
        mock_load_texts.return_value = ["Текст для тестирования", "Тестирование TF-IDF"]
        mock_preprocess_texts.return_value = ['текст тестирование', 'тестирование tf idf']
        mock_create_tfidf.return_value = (MagicMock(), MagicMock())

        # Успешный вызов
        save_tfidf_model_and_index()
        output = mock_stdout.getvalue()

        # Проверка вызовов
        mock_load_texts.assert_called_once_with(DATA_FOLDER, "text")
        mock_preprocess_texts.assert_called_once_with(["Текст для тестирования", "Тестирование TF-IDF"])
        mock_create_tfidf.assert_called_once_with(['текст тестирование', 'тестирование tf idf'])
        self.assertEqual(mock_pickle_dump.call_count, 3)
        self.assertIn("TF-IDF индекс успешно создан и сохранён", output)

    @patch("app.text_search.create_tfidf.load_texts_from_folder")
    @patch("sys.stdout", new_callable=StringIO)
    def test_save_tfidf_model_and_index_no_texts(self, mock_stdout, mock_load_texts):
        """Тест обработки ошибки при отсутствии текстов"""
        mock_load_texts.side_effect = FileNotFoundError("Папка не найдена")
        save_tfidf_model_and_index()
        output = mock_stdout.getvalue()
        self.assertIn("Ошибка: Папка не найдена", output)


if __name__ == "__main__":
    unittest.main()
