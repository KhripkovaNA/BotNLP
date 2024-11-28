import spacy

# Загрузка русской языковой модели spaCy
nlp = spacy.load("ru_core_news_sm")


def preprocess_text(text):
    """
    Обрабатывает текст: токенизация, удаление стоп-слов, приведение к нижнему регистру и лемматизация.
    :param text: строка с текстом для обработки
    :return: список обработанных слов
    """
    # Обработка текста через spaCy
    doc = nlp(text)

    # Лемматизация, удаление стоп-слов и неалфавитных символов
    processed_tokens = [
        token.lemma_.lower() for token in doc
        if token.is_alpha and not token.is_stop
    ]

    return processed_tokens
