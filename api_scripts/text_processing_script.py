import asyncio
import httpx

BASE_URL = "http://127.0.0.1:8000"


async def preprocess_text(text: str):
    """Отправляет текст на обработку к API и возвращает результат"""
    endpoint = f"{BASE_URL}/api/preprocess"
    payload = {"text": text}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()  # Поднимает исключение для статусов 4xx/5xx
            return response.json()
        except httpx.RequestError as e:
            print(f"Ошибка при попытке соединения с API: {e}")
        except httpx.HTTPStatusError as e:
            print(f"Ошибка HTTP статуса: {e.response.status_code}, {e.response.text}")

if __name__ == "__main__":
    text_to_process = "Какие товары пользователи считают удобными в использовании?"
    print("Текст для обработки: ", text_to_process)
    result = asyncio.run(preprocess_text(text_to_process))
    if result:
        print("Обработанный текст:", result.get("processed_text"))
