import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import pickle
from app.text_processing.service import preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / "data"
TFIDF_FOLDER = PROJECT_ROOT / "tfidf"


def load_texts_from_folder(folder_path: Path, key: str = None) -> List[str]:
    """
    Загружает тексты из всех .json файлов в указанной папке
    :param folder_path: Путь к папке с файлами .json
    :param key: Ключ для извлечения текстов из объектов JSON (если данные представлены в виде словаря)
    :return: Список строк, каждая из которых — содержимое текстового поля из JSON
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Папка '{folder_path}' не найдена")

    texts = []
    for file_path in folder_path.glob("*.json"):
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

            if isinstance(data, list):
                # Если JSON содержит список объектов
                texts.extend(item[key] if key else str(item) for item in data if
                             isinstance(item, dict) and (key is None or key in item))
            elif isinstance(data, dict):
                # Если JSON является словарём
                if key:
                    if key in data:
                        texts.append(data[key])
                else:
                    texts.append(str(data))
            else:
                raise ValueError(
                    f"Формат данных в файле {file_path} не поддерживается (ожидается список или словарь)")

    if not texts:
        raise ValueError(f"В папке '{folder_path}' нет подходящих файлов .json или они пусты")

    return texts


def preprocess_texts(raw_texts: List[str]) -> List[str]:
    """
    Обрабатывает список текстов
    :param raw_texts: исходные тексты
    :return: обработанные тексты
    """
    return [" ".join(preprocess_text(text)) for text in raw_texts]


def create_tfidf_model_and_index(processed_texts: List[str]) -> Tuple[TfidfVectorizer, List[str]]:
    """
    Создаёт TF-IDF индекс из обработанных текстов
    :param processed_texts: список обработанных текстов
    :return: модель TF-IDF и матрица текста
    """
    if not processed_texts:
        raise ValueError("Обработанные тексты пусты. Создание TF-IDF невозможно")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts).toarray()
    return vectorizer, tfidf_matrix


def save_tfidf_model_and_index(data_folder, tfidf_folder):
    try:
        # Загрузка текстов
        print("Загрузка текстов...")
        texts = load_texts_from_folder(data_folder, "text")
        print(f"Загружено {len(texts)} текстов")

        # Предобработка текстов
        print("Предобработка текстов...")
        processed_texts = preprocess_texts(texts)

        print("Создание TF-IDF индекса...")
        vectorizer, tfidf_matrix = create_tfidf_model_and_index(processed_texts)

        # Сохранение модели и индекса
        print("Сохранение модели и матрицы...")
        tfidf_folder.mkdir(parents=True, exist_ok=True)

        with open(tfidf_folder / "texts.pkl", "wb") as texts_file:
            pickle.dump(texts, texts_file)
        with open(tfidf_folder / "tfidf_model.pkl", "wb") as model_file:
            pickle.dump(vectorizer, model_file)
        with open(tfidf_folder / "tfidf_matrix.pkl", "wb") as matrix_file:
            pickle.dump(tfidf_matrix, matrix_file)

        print("TF-IDF индекс успешно создан и сохранён")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    save_tfidf_model_and_index(DATA_FOLDER, TFIDF_FOLDER)
