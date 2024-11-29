import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from app.text_processing.service import preprocess_text
from app.text_search.create_tfidf_index import TFIDF_FOLDER

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    if not isinstance(query, str):
        raise ValueError("Запрос должен быть строкой")

    # Обработка запроса
    processed_query = " ".join(preprocess_text(query))

    # Преобразование запроса в вектор
    query_vector = vectorizer.transform([processed_query])

    # Вычисление косинусного сходства
    similarities = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()

    # Получение индексов топ-3 результатов
    top_indices = similarities.argsort()[-3:][::-1]

    # Возврат текстов и их релевантности
    results = [(texts[i], similarities[i]) for i in top_indices if similarities[i] > 0]
    return results


def get_relevant_texts(query: str) -> List[Tuple[str, float]]:
    """
    Возвращает топ-3 релевантных текста для запроса
    :param query: Текст запроса
    :return: Список из 3 текстов и их релевантности
    """
    models_folder = PROJECT_ROOT / TFIDF_FOLDER
    model_path = models_folder / "tfidf_model.pkl"
    matrix_path = models_folder / "tfidf_matrix.pkl"
    texts_path = models_folder / "texts.pkl"

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
