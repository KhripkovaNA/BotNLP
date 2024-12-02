from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.text_processing.router import router as processing_router
from app.text_search.router import router as text_search_router

# Инициализация FastAPI приложения
app = FastAPI(
    title="Поиск по тексту"
)

# Подключение роутеров
app.include_router(processing_router, prefix="/api", tags=["Text Processing"])
app.include_router(text_search_router, prefix="/api", tags=["Text Search"])


# Обработчик исключений Pydantic валидации
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = " -> ".join(map(str, error["loc"]))
        msg = error["msg"]
        errors.append(f"Ошибка в поле '{field}': {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Ошибка валидации данных",
            "errors": errors,
        },
    )


# Обработчик для всех исключений, унаследованных от HTTPException
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
