import unittest
from app.text_processing.service import preprocess_text


class TestLibrary(unittest.TestCase):
    def test_process_text(self):
        """Тест обработки текста"""
        original_text = "Привет! Как дела? Это пример текста для обработки с помощью NLP"
        processed_words = ['привет', 'дело', 'пример', 'текст', 'обработка', 'помощь', 'nlp']
        output = preprocess_text(original_text)
        self.assertEqual(output, processed_words)


if __name__ == '__main__':
    unittest.main()
