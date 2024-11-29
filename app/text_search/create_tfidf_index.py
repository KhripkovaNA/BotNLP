from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
import pickle
from app.text_processing.service import preprocess_text

DATA_FOLDER = "data"
TFIDF_FOLDER = "tfidf"


def load_texts_from_folder(folder_path: Path) -> List[str]:
    """
    Загружает тексты из всех .txt файлов в указанной папке
    :param folder_path: Путь к папке с файлами .txt
    :return: Список строк, каждая из которых — содержимое файла
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Папка '{folder_path}' не найдена")

    texts = []
    for file_path in folder_path.glob("*.txt"):  # Ищем все .txt файлы
        with file_path.open("r", encoding="utf-8") as file:
            texts.append(file.read())

    if not texts:
        raise ValueError(f"В папке '{folder_path}' нет файлов .txt")

    return texts


def preprocess_texts(raw_texts: List[str]) -> List[str]:
    """
    Обрабатывает список текстов.
    :param raw_texts: исходные тексты
    :return: обработанные тексты
    """
    return [" ".join(preprocess_text(text)) for text in raw_texts]


def create_tfidf_index(processed_texts: List[str]) -> Tuple[TfidfVectorizer, List[str]]:
    """
    Создаёт TF-IDF индекс из обработанных текстов.
    :param processed_texts: список обработанных текстов
    :return: модель TF-IDF и матрица текста
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    return vectorizer, tfidf_matrix


def save_tfidf_models():
    project_root = Path(__file__).resolve().parents[2]  # Путь к корневой папке
    data_folder = project_root / DATA_FOLDER

    try:
        # Загрузка текстов
        texts = load_texts_from_folder(data_folder)
        print(f"Загружено {len(texts)} текстов")

        # Предобработка текстов
        processed_texts = preprocess_texts(texts)

        # Создание индекса
        vectorizer, tfidf_matrix = create_tfidf_index(processed_texts)

        # Сохранение модели и индекса
        models_folder = project_root / TFIDF_FOLDER
        models_folder.mkdir(parents=True, exist_ok=True)

        with open(models_folder / "texts.pkl", "wb") as texts_file:
            pickle.dump(texts, texts_file)
        with open(models_folder / "tfidf_model.pkl", "wb") as model_file:
            pickle.dump(vectorizer, model_file)
        with open(models_folder / "tfidf_matrix.pkl", "wb") as matrix_file:
            pickle.dump(tfidf_matrix, matrix_file)

        print("TF-IDF индекс успешно создан и сохранён")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    save_tfidf_models()
