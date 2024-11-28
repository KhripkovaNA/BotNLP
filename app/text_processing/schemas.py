# Модель для валидации входных данных
from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str
