from fastapi import HTTPException, status


class AppBaseException(HTTPException):
    # Значения по умолчанию
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Внутренняя ошибка сервера"

    def __init__(self):
        super().__init__(status_code=self.status_code, detail=self.detail)
