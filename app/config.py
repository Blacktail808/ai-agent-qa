"""
Настройки приложения.

Все настройки можно задать в файле .env (образец — в .env.example).
Так можно переключаться между разными LLM (Ollama, OpenRouter, Yandex Cloud,
OpenAI) НЕ меняя код — достаточно поменять несколько строк в .env.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Базовый URL OpenAI-совместимого API.
    #   Ollama:       http://localhost:11434/v1
    #   vLLM:         http://localhost:8000/v1  (если vLLM не конфликтует с нашим портом)
    #   OpenRouter:   https://openrouter.ai/api/v1
    #   Yandex Cloud: https://foundationmodels.api.cloud.yandex.net/v1
    #   OpenAI:       https://api.openai.com/v1
    llm_base_url: str = "http://localhost:11434/v1"

    # API-ключ. Для локального Ollama/vLLM подойдёт любое значение,
    # например "not-needed". Для внешних сервисов — ключ из личного кабинета.
    llm_api_key: str = "not-needed"

    # Название модели. Для Ollama — любая установленная модель (команда `ollama list`).
    llm_model: str = "qwen3:8b"

    # «Температура» — насколько «творческими» и разнообразными будут ответы модели.
    # 0.0 — максимально строго и предсказуемо, 1.0 — максимально творчески.
    # Для оценки качества диалогов нужна низкая температура (0.0–0.3).
    llm_temperature: float = 0.2

    # Сколько секунд ждать ответ от модели.
    llm_timeout_seconds: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Единый объект с настройками: импортируем его в других файлах.
settings = Settings()
