from typing import List
from . import nlp


def preprocess_text(text: str) -> List[str]:
    """
    Обрабатывает текст: токенизация, удаление стоп-слов, приведение к нижнему регистру и лемматизация.
    :param text: строка с текстом для обработки
    :return: список обработанных слов
    """
    if not isinstance(text, str):
        raise ValueError("Входной текст должен быть строкой")

    if not text.strip():  # Если текст пустой, вернуть пустой список
        return []

    # Обработка текста через spaCy
    doc = nlp(text)

    # Лемматизация, удаление стоп-слов и неалфавитных символов
    processed_tokens = [
        token.lemma_.lower() for token in doc
        if token.is_alpha and not token.is_stop
    ]

    return processed_tokens
