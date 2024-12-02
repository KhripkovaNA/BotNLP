import unittest
from app.text_processing.service import preprocess_text


class TestTextProcessingService(unittest.TestCase):
    def test_process_text_normal(self):
        """Тест нормальной обработки текста"""
        original_text = "Привет! Как дела? Это пример текста для обработки с помощью NLP"
        processed_words = ['привет', 'дело', 'пример', 'текст', 'обработка', 'помощь', 'nlp']
        output = preprocess_text(original_text)
        self.assertEqual(output, processed_words)

    def test_process_empty_text(self):
        """Тест обработки пустого текста"""
        original_text = ""
        processed_words = []
        output = preprocess_text(original_text)
        self.assertEqual(output, processed_words)

    def test_process_non_alpha_text(self):
        """Тест обработки текста с неалфавитными символами"""
        original_text = "12345 !!!! ???"
        processed_words = []
        output = preprocess_text(original_text)
        self.assertEqual(output, processed_words)

    def test_process_invalid_type(self):
        """Тест обработки некорректного типа данных"""
        with self.assertRaises(ValueError):
            preprocess_text(12345)


if __name__ == '__main__':
    unittest.main()
