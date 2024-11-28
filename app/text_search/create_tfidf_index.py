import os
from typing import List


def load_texts_from_data_folder(folder_path: str) -> List[str]:
    """
    Загружает тексты из всех .txt файлов в указанной папке
    :param folder_path: Путь к папке с файлами .txt
    :return: Список строк, каждая из которых — содержимое файла
    """
    texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Проверяем, что это файл с расширением .txt
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())

    return texts


if __name__ == "__main__":
    folder = "./data"  # Папка с текстовыми файлами
    all_texts = load_texts_from_data_folder(folder)
    print(f"Загружено {len(all_texts)} текстов.")
    for idx, text in enumerate(all_texts, 1):
        print(f"--- Текст {idx} ---\n{text[:100]}...")  # Печатаем первые 100 символов каждого текста