from fastapi import APIRouter
from .schemas import TextRequest
from .service import preprocess_text

router = APIRouter()


@router.post("/preprocess", summary="Обработка текста", description="Возвращает обработанные слова из текста")
async def preprocess_endpoint(request: TextRequest):
    """Эндпоинт для обработки текста"""
    processed_text = preprocess_text(request.text)
    return {"processed_text": processed_text}
