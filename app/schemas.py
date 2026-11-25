"""
Модели данных (схемы Pydantic).

Pydantic — библиотека Python, которая описывает «как выглядят правильные
данные» и автоматически проверяет входящие и исходящие данные по этому описанию.

Именно благодаря этим моделям:
1) FastAPI проверит запрос клиента (например, что диалог — список сообщений);
2) мы проверим ответ модели;
3) наш API всегда отдаёт валидный JSON.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Одно сообщение анализируемого диалога."""

    role: Literal["user", "assistant", "system"] = Field(
        description="Кто сказал: user (пользователь), assistant (AI-агент) или system (инструкция)"
    )
    content: str = Field(description="Текст сообщения")


class AnalyzeRequest(BaseModel):
    """Тело запроса: диалог, который нужно оценить."""

    dialog: list[ChatMessage] = Field(
        min_length=2,
        description="Диалог между AI-агентом (assistant) и пользователем (user)",
    )
    agent_goal: str | None = Field(
        default=None,
        description="Цель, которую должен был достичь AI-агент (необязательно)",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Температура генерации для этого запроса (0.0–1.0, необязательно)",
    )


class CriterionScore(BaseModel):
    """Оценка по одному критерию."""

    name: str = Field(description="Название критерия")
    score: int = Field(ge=0, le=10, description="Оценка от 0 до 10")
    comment: str = Field(description="Почему поставлена такая оценка")


class AnalyzeResult(BaseModel):
    """Ответ API: отчёт о качестве диалога (валидный JSON)."""

    overall_score: int = Field(ge=0, le=100, description="Итоговая оценка от 0 до 100")
    verdict: Literal["excellent", "good", "needs_improvement", "poor"] = Field(
        description="Итоговый вердикт"
    )
    criteria: list[CriterionScore] = Field(
        min_length=3,
        max_length=8,
        description="Оценки по отдельным критериям (релевантность, вежливость и т.д.)",
    )
    issues: list[str] = Field(description="Найденные проблемы диалога (может быть пустым списком)")
    recommendations: list[str] = Field(
        min_length=1,
        description="Как улучшить AI-агента",
    )
    hallucination_detected: bool = Field(
        description="Выдумывал ли агент несуществующие факты (галлюцинировал)"
    )
