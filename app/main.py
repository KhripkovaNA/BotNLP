from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from app.text_processing.router import router as processing_router

# Инициализация FastAPI приложения
app = FastAPI(
    title="Мои Задачи"
)

# Подключение роутеров
app.include_router(processing_router, prefix="/api", tags=["Text Processing"])


# Глобальный обработчик для всех исключений, унаследованных от HTTPException
@app.exception_handler(HTTPException)
async def app_base_exception_handler(request: Request, e: HTTPException):

    return JSONResponse(
        status_code=e.status_code,
        content={"detail": e.detail}
    )


# Обработчик для всех других необработанных ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, e: Exception):

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Внутренняя ошибка сервера"}
    )
