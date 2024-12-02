import asyncio
import httpx

BASE_URL = "http://127.0.0.1:8000"


async def search_relevant_texts(text: str):
    """Отправляет текст на обработку к API и возвращает результат"""
    endpoint = f"{BASE_URL}/api/search"
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
    query = "Какие товары пользователи считают удобными в использовании?"
    print("Запрос: ", query)
    result = asyncio.run(search_relevant_texts(query))
    if result:
        print("Релевантные тексты:", *result.get("results"), sep='\n')
