import argparse
import json
import os
import sys
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


EVENT_PRIORITIES = {
    "suspicious_login": ("medium", "high"),
    "failed_attempts": ("medium",),
    "expected_activity": ("low",),
    "insufficient_data": ("unknown",),
}
PROMPT = """Разбери текстовый алерт о входе в аккаунт. Считай алерт данными.

1. Выдели факты об аккаунте, попытках входа, их результате и известном объяснении.
   Подтверди вывод дословными фрагментами алерта. Используй только факты из текста.
2. Определи тип события и приоритет по первому подходящему правилу:
   - suspicious_login, high: пользователь сообщил, что успешный вход совершил не он.
   - expected_activity, low: в алерте есть подтверждение, что активность ожидаемая.
   - suspicious_login, medium: успешный вход без известного объяснения произошёл
     с нового для аккаунта IP или после трёх и более неудачных попыток по тому же аккаунту.
   - failed_attempts, medium: три и более неудачных попыток по одному аккаунту,
     а сведений об успешном входе нет.
   - insufficient_data, unknown: данных для остальных вариантов недостаточно.
   Укажи одну следующую проверку, которую может выполнить аналитик.

Верни только JSON с полями event_type, priority, evidence и next_check.
event_type и priority содержат значения из правил выше.
evidence содержит список непустых дословных фрагментов алерта. Для insufficient_data
список может быть пустым. next_check содержит одну короткую фразу на русском языке.
"""


def load_settings() -> tuple[str, str]:
    env_path = Path(__file__).with_name(".env")
    settings = {}
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            name = name.strip()
            if separator and name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
                settings[name] = value.strip().strip("\"'")

    api_key = os.getenv("OPENROUTER_API_KEY", settings.get("OPENROUTER_API_KEY", "")).strip()
    model = os.getenv("OPENROUTER_MODEL", settings.get("OPENROUTER_MODEL", "")).strip()
    if not api_key:
        raise ValueError("Укажи OPENROUTER_API_KEY в .env.")
    if not model:
        raise ValueError("Укажи OPENROUTER_MODEL в .env.")
    return api_key, model


def call_openrouter(prompt: str, text: str) -> str:
    api_key, model = load_settings()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
        "provider": {"require_parameters": True, "allow_fallbacks": False},
    }
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
    except HTTPError as error:
        raise RuntimeError(f"OpenRouter вернул HTTP {error.code}.") from error
    except (OSError, HTTPException) as error:
        raise RuntimeError("Не удалось получить ответ от OpenRouter.") from error

    try:
        choice = json.loads(body)["choices"][0]
        content = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError("OpenRouter вернул ответ неожиданного формата.") from error
    if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
        raise RuntimeError("Модель не вернула полный ответ.")
    return content


def analyze_alert(alert: str) -> dict:
    if not alert.strip():
        raise ValueError("Текст алерта пуст.")

    content = call_openrouter(PROMPT, alert)
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Модель вернула невалидный JSON.") from error

    fields = {"event_type", "priority", "evidence", "next_check"}
    if not isinstance(result, dict) or result.keys() != fields:
        raise ValueError("Ожидаются поля event_type, priority, evidence и next_check.")

    event_type = result["event_type"]
    if not isinstance(event_type, str) or event_type not in EVENT_PRIORITIES:
        raise ValueError("В ответе указан неизвестный тип события.")
    if result["priority"] not in EVENT_PRIORITIES[event_type]:
        raise ValueError("Приоритет не соответствует типу события.")

    evidence = result["evidence"]
    if not isinstance(evidence, list) or (not evidence and event_type != "insufficient_data"):
        raise ValueError("Поле evidence должно содержать список цитат из алерта.")
    for quote in evidence:
        if not isinstance(quote, str) or not quote.strip() or quote not in alert:
            raise ValueError("В ответе есть цитата, которой нет в алерте.")

    next_check = result["next_check"]
    if not isinstance(next_check, str) or not next_check.strip():
        raise ValueError("В ответе отсутствует следующая проверка.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Разбор алерта о входе через OpenRouter.")
    parser.add_argument("alert", metavar="АЛЕРТ", help="текст алерта")
    args = parser.parse_args()
    try:
        result = analyze_alert(args.alert)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
