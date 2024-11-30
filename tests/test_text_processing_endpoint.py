import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestPreprocessEndpoint(unittest.TestCase):
    def setUp(self):
        """Создание тестовой среды"""
        self.client = TestClient(app)

    def test_preprocess_valid_text(self):
        """Тест обработки корректного текста"""
        response = self.client.post(
            "/api/preprocess",
            json={"text": "Привет! Как дела? Это пример текста для обработки."}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("processed_text", response.json())
        self.assertEqual(
            response.json()["processed_text"],
            ['привет', 'дело', 'пример', 'текст', 'обработка']
        )

    def test_preprocess_empty_text(self):
        """Тест обработки пустого текста"""
        response = self.client.post("/api/preprocess", json={"text": ""})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["processed_text"], [])

    def test_preprocess_invalid_data(self):
        """Тест обработки некорректного типа данных"""
        response = self.client.post("/api/preprocess", json={"text": 12345})
        self.assertEqual(response.status_code, 422)  # FastAPI вернёт ошибку валидации

    def test_preprocess_missing_field(self):
        """Тест отсутствия обязательного поля"""
        response = self.client.post("/api/preprocess", json={})
        self.assertEqual(response.status_code, 422)  # FastAPI вернёт ошибку валидации


if __name__ == '__main__':
    unittest.main()
