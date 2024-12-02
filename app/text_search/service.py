import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.text_processing.service import preprocess_text


# Загрузка модели и индекса
def load_tfidf_model_and_index(model_path: Path, matrix_path: Path) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Загружает модель TF-IDF и матрицу индекса из файлов
    :param model_path: Путь к файлу с моделью TF-IDF
    :param matrix_path: Путь к файлу с матрицей TF-IDF
    :return: Модель TF-IDF и матрица индекса
    """
    with open(model_path, "rb") as model_file:
        vectorizer = pickle.load(model_file)
    with open(matrix_path, "rb") as matrix_file:
        tfidf_matrix = pickle.load(matrix_file)
    return vectorizer, tfidf_matrix


# Поиск релевантных текстов
def search_texts(
        query: str, vectorizer: TfidfVectorizer, tfidf_matrix: np.ndarray, texts: List[str]
) -> List[Tuple[str, float]]:
    """
    Ищет 3 наиболее релевантных текста для запроса
    :param query: Текст запроса
    :param vectorizer: Модель TF-IDF
    :param tfidf_matrix: Матрица TF-IDF
    :param texts: Исходные тексты, соответствующие индексу
    :return: Список из 3 текстов и их релевантности
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Запрос должен быть непустой строкой")

    # Обработка запроса
    processed_query = " ".join(preprocess_text(query))

    # Преобразование запроса в вектор
    query_vector = vectorizer.transform([processed_query]).toarray()

    # Вычисление косинусного сходства
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Получение индексов топ-3 результатов
    top_indices = similarities.argsort()[-3:][::-1]

    # Возврат текстов и их релевантности
    results = [(texts[i], similarities[i]) for i in top_indices]
    return results


# Получение релевантных текстов
def get_relevant_texts(query: str, tfidf_folder: Path) -> List[Tuple[str, float]]:
    """
    Возвращает топ-3 релевантных текста для запроса
    :param query: Текст запроса
    :param tfidf_folder: Путь к папке с TF-IDF моделью и матрицей
    :return: Список из 3 текстов и их релевантности
    """
    model_path = tfidf_folder / "tfidf_model.pkl"
    matrix_path = tfidf_folder / "tfidf_matrix.pkl"
    texts_path = tfidf_folder / "texts.pkl"

    # Проверяем наличие всех необходимых файлов
    missing_files = []
    for file_path, description in [(model_path, "модель TF-IDF"),
                                   (matrix_path, "матрица TF-IDF"),
                                   (texts_path, "исходные тексты")]:
        if not file_path.exists():
            missing_files.append(f"{description} ({file_path})")

    if missing_files:
        raise FileNotFoundError(f"Следующие файлы не найдены: {', '.join(missing_files)}")

    vectorizer, tfidf_matrix = load_tfidf_model_and_index(model_path, matrix_path)
    with open(texts_path, "rb") as texts_file:
        texts = pickle.load(texts_file)
    results = search_texts(query, vectorizer, tfidf_matrix, texts)

    return results
