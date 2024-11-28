from fastapi import APIRouter
from .service import preprocess_text

router = APIRouter()


@router.post("/preprocess")
async def preprocess_endpoint(text: str):
    """
    Эндпоинт для обработки текста
    """
    processed_text = preprocess_text(text)
    return {"processed_text": processed_text}
