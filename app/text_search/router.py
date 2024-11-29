from typing import Optional
from fastapi import APIRouter, HTTPException
from app.text_processing.schemas import TextRequest
from app.text_search.service import get_relevant_texts

router = APIRouter()


@router.post("/search", summary="Поиск по тексту", description="Возвращает 3 наиболее релевантных текста для запроса")
async def search_endpoint(request: Optional[TextRequest]):
    """Эндпоинт для поиска текста"""
    try:
        query = request.text
        results = get_relevant_texts(query)
        return {"query": query, "results": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
