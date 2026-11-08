import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter, sleep

from main import analyze_alert, analyze_workflow, load_settings


CASES = [
    (
        "login_new_ip",
        "Для user01 зафиксировано 5 неудачных попыток входа. "
        "Затем выполнен успешный вход в тот же аккаунт с нового IP.",
        "suspicious_login", "medium",
    ),
    (
        "login_denied",
        "В 02:10 выполнен успешный вход в user02. Владелец аккаунта сообщил, "
        "что в это время не входил в аккаунт.",
        "suspicious_login", "high",
    ),
    (
        "login_failures",
        "Для user03 зарегистрировано 7 неудачных попыток входа за 5 минут. "
        "Успешных входов в журнале нет.",
        "failed_attempts", "medium",
    ),
    (
        "login_vpn",
        "Для user04 отмечено 6 неудачных попыток входа, затем успешный вход с нового IP. "
        "Владелец подтвердил, что сам вошёл через корпоративный VPN после ошибок в пароле.",
        "expected_activity", "low",
    ),
    (
        "phishing_email",
        "Письмо выдаёт себя за службу поддержки и просит ввести корпоративный пароль "
        "на сайте helpdesk.example.invalid. Получатель переслал письмо и сообщил, что не вводил пароль.",
        "phishing", "medium",
    ),
    (
        "phishing_password",
        "Письмо от имени ИТ-службы направило сотрудника на поддельную страницу входа. "
        "Сотрудник подтвердил, что ввёл на ней корпоративный пароль.",
        "phishing", "high",
    ),
    (
        "phishing_drill",
        "Письмо имитирует запрос пароля от ИТ-службы. Сотрудник открыл страницу из письма. "
        "Команда ИБ подтвердила согласованную учебную рассылку. Страница не принимала пароли.",
        "expected_activity", "low",
    ),
    (
        "malware_blocked",
        "Антивирус подтвердил, что файл invoice.exe вредоносный. Файл помещён в карантин, "
        "запуск заблокирован.",
        "malware", "medium",
    ),
    (
        "malware_executed",
        "Антивирус подтвердил вредоносность файла update.exe на ws-07. "
        "Журнал процессов подтверждает, что этот файл запустился до обнаружения.",
        "malware", "high",
    ),
    (
        "malware_unknown",
        "Система мониторинга отметила необычное имя файла report.exe. "
        "В приложенном отчёте отсутствуют вердикт антивируса и сведения о запуске.",
        "insufficient_data", "unknown",
    ),
]


def run_case(analyze, alert: str, event_type: str, priority: str) -> dict:
    request_tokens = []
    result = None
    error = None
    started = perf_counter()
    try:
        result = analyze(alert, request_tokens=request_tokens)
    except (ValueError, RuntimeError, OSError) as exc:
        error = str(exc)
    elapsed = perf_counter() - started
    return {
        "result": result,
        "error": error,
        "correct": (
            result is not None
            and result["event_type"] == event_type
            and result["priority"] == priority
        ),
        "seconds": round(elapsed, 4),
        "calls": len(request_tokens),
        "tokens": None if None in request_tokens else sum(request_tokens),
        "tokens_by_call": request_tokens,
    }


def summarize(runs: list[dict]) -> dict:
    tokens = [run["tokens"] for run in runs if run["tokens"] is not None]
    return {
        "success_rate": mean(run["result"] is not None for run in runs),
        "accuracy": mean(run["correct"] for run in runs),
        "mean_seconds": round(mean(run["seconds"] for run in runs), 3),
        "mean_calls": mean(run["calls"] for run in runs),
        "mean_tokens": round(mean(tokens), 1) if tokens else None,
        "token_samples": len(tokens),
    }


def main() -> None:
    _, model = load_settings()
    started_at = datetime.now(timezone.utc).isoformat()
    modes = {"one_shot": analyze_alert, "workflow": analyze_workflow}
    records = []

    for index, (case_id, alert, event_type, priority) in enumerate(CASES):
        if index:
            sleep(12)
        order = list(modes) if index % 2 == 0 else list(reversed(modes))
        runs = {mode: run_case(modes[mode], alert, event_type, priority) for mode in order}
        records.append({
            "id": case_id,
            "alert": alert,
            "expected": {"event_type": event_type, "priority": priority},
            "runs": runs,
        })
        print(f"{index + 1}/{len(CASES)} {case_id}", flush=True)

    summary = {mode: summarize([record["runs"][mode] for record in records]) for mode in modes}
    report = {"model": model, "started_at": started_at, "summary": summary, "cases": records}
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
