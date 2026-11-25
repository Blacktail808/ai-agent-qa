"""
Сам HTTP API (FastAPI).

FastAPI — фреймворк, который превращает обычные функции Python в веб-адреса
(эндпоинты): вы описываете, что происходит при запросе по адресу X, а фреймворк
берёт на себя остальное — HTTP, проверку запроса по схемам Pydantic,
превращение ответа в JSON и даже автоматическую интерактивную документацию
по адресу /docs (можно тестировать API прямо из браузера).

Запуск:   uvicorn app.main:app --reload --port 8000
Документация: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException

from .config import settings
from .llm import LLMResponseError, LLMUnavailableError, analyze_dialog_quality
from .schemas import AnalyzeRequest, AnalyzeResult

app = FastAPI(
    title="Цифровой сотрудник — анализатор качества диалогов AI-агента",
    version="0.1.0",
    description=(
        "Прототип API: принимает диалог AI-агента с пользователем, отправляет его "
        "в LLM через OpenAI-совместимый API с заданной температурой и возвращает "
        "валидный JSON-отчёт о качестве (структурированный вывод)."
    ),
)


@app.get("/", summary="Информация о сервисе")
def root():
    return {
        "service": "Цифровой сотрудник — анализатор качества диалогов AI-агента",
        "status": "running",
        "model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "default_temperature": settings.llm_temperature,
        "docs": "/docs",
        "analyze_endpoint": "POST /analyze_dialog",
    }


@app.get("/health", summary="Проверка живости сервиса")
def health():
    return {"status": "ok"}


@app.post(
    "/analyze_dialog",
    response_model=AnalyzeResult,
    summary="Оценить качество диалога AI-агента",
)
async def analyze_dialog(request: AnalyzeRequest) -> AnalyzeResult:
    """Принимает диалог, просит LLM оценить его и возвращает проверенный JSON-отчёт.

    Пример тела запроса — в файле examples/sample_dialog.json
    (или нажмите "Try it out" на странице /docs).
    """
    try:
        return await analyze_dialog_quality(request)
    except LLMUnavailableError as error:
        # 502 — наш сервис работает, но модель за ним недоступна.
        raise HTTPException(status_code=502, detail=str(error)) from error
    except LLMResponseError as error:
        # 502 — модель вернула невалидные данные; клиенту мы их не отдаём.
        raise HTTPException(status_code=502, detail=str(error)) from error
