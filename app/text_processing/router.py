from typing import Optional
from fastapi import APIRouter, HTTPException
from .schemas import TextRequest
from .service import preprocess_text

router = APIRouter()


@router.post("/preprocess", summary="Обработка текста", description="Возвращает обработанные слова из текста")
async def preprocess_endpoint(request: Optional[TextRequest]):
    """Эндпоинт для обработки текста"""
    try:
        processed_text = preprocess_text(request.text)
        return {"processed_text": processed_text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
