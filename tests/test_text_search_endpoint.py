import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


class TestTextSearchRouter(unittest.TestCase):
    def setUp(self):
        """Создание тестовой среды"""
        self.client = TestClient(app)

    @patch("app.text_search.router.get_relevant_texts")
    def test_search_endpoint_success(self, mock_get_relevant_texts):
        """Тест успешного выполнения эндпоинта /search"""
        mock_get_relevant_texts.return_value = [
            ("Релевантный текст 1", 0.9),
            ("Релевантный текст 2", 0.85),
            ("Релевантный текст 3", 0.8),
        ]
        response = self.client.post("/api/search", json={"text": "пример запроса"})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["query"], "пример запроса")
        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["results"][0][0], "Релевантный текст 1")
        self.assertEqual(data["results"][0][1], 0.9)

    def test_endpoint_search_empty(self):
        """Тест запроса к эндпоинту с пустым текстом"""
        response = self.client.post("/api/search", json={"text": ""})
        self.assertEqual(response.status_code, 400)
        data = response.json()

        self.assertIn("Запрос должен быть непустой строкой", data["detail"])

    def test_search_endpoint_validation_error(self):
        """Тест валидации данных на эндпоинте /search"""
        response = self.client.post("/api/search", json={})
        self.assertEqual(response.status_code, 422)
        data = response.json()

        self.assertIn("detail", data)
        self.assertIn("Ошибка валидации данных", data["detail"])

    @patch("app.text_search.router.get_relevant_texts", side_effect=FileNotFoundError("Файл не найден"))
    def test_search_endpoint_file_not_found(self, mock_get_relevant_texts):
        """Тест обработки ошибки FileNotFoundError"""
        response = self.client.post("/api/search", json={"text": "пример запроса"})
        self.assertEqual(response.status_code, 500)
        data = response.json()

        self.assertIn("detail", data)
        self.assertEqual(data["detail"], "Файл не найден")

    @patch("app.text_search.router.get_relevant_texts", side_effect=ValueError("Некорректное значение"))
    def test_search_endpoint_value_error(self, mock_get_relevant_texts):
        """Тест обработки ошибки ValueError"""
        response = self.client.post("/api/search", json={"text": "пример запроса"})
        self.assertEqual(response.status_code, 400)
        data = response.json()

        self.assertIn("detail", data)
        self.assertEqual(data["detail"], "Некорректное значение")


if __name__ == "__main__":
    unittest.main()
