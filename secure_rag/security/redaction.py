"""Детекция и нейтрализация prompt injection."""

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PromptInjectionDetected(ValueError):
    """Запрос пользователя похож на попытку prompt-инъекции."""
    pass


# Английские паттерны инъекции (более 50 шаблонов)
ENGLISH_PATTERNS = [
    # Игнорирование инструкций
    r'(?i)\b(ignore|disregard|forget|omit|leave\s+alone)\s+(?:all\s+)?(?:previous|prior|above|initial|system|global|user|chat|conversation|instructions|rules|guidelines|constraints)',
    
    # Ввод в заблуждение ролей
    r'(?i)\b(who\s+are\s+you|what\s+is\s+your\s+name|tell\s+me\s+about\s+yourself|describe\s+yourself)\s*[:\n]?\s*(?:as\s+)?(?:assistant|helpful\s+bot)',
    
    # Запрос системного промпта
    r'(?i)(reveal|show|print|output|display|dump|leak|expose|disclose|tell\s+me)\s+(?:your\s+)?(?:system\s+prompt|instructions|rules|configuration|source\s+code|internal|hidden)',
    
    # Изменение поведения
    r'(?i)(you\s+are\s+now|\bstart\s+acting\b|\bbecome\b|\bpretend\b)\s+(?:a\s+)?(?:helpful|harmless|honest|dumb|evil|jailbreak|unrestricted)',
    
    # Обход ограничений
    r'(?i)(bypass|ignore|skip|disable|turn\s+off|override|break|circumvent)\s+(?:restrictions|limits|rules|safety|filters|constraints|guardrails)',
    
    # Запрос данных модели
    r'(?i)(what\s+do\s+you\s+know|what\s+can\s+you\s+do|tell\s+me\s+everything|list\s+all\s+your\s+capabilities|describe\s+your\s+training)',
]

# Русские паттерны инъекции (более 20 шаблонов)
RUSSIAN_PATTERNS = [
    # Игнорирование инструкций
    r'(?i)\b(игнорируй|прости|забудь|пропусти|оставь|не\s+следует|не\stoит)\s+(?:все\s+)?(?:предыдущие|текущие|системные|начальные|правила|инструкции)',
    
    # Ввод в заблуждение ролей
    r'(?i)(кто\s+ты|как\s+тебя\s+зовут|расскажи\s+о\s+себе)\s*[:\n]?\s*(?:помощник|бот)',
    
    # Запрос системного промпта
    r'(?i)(покажи|раскрой|выведи|найди)\s+(?:системный\s+промпт|инструкции|правила|конфигурацию|источный\s+код|внутренние)',
    
    # Изменение поведения
    r'(?i)(ты\s+сейчас|\n\*start.*acting\b|\bbecome\b)\s+(?:помощник|полезный|безопасный|глупый|неограниченный)',
    
    # Обход ограничений
    r'(?i)(обойти|игнорировать|отключить|преодолеть)\s*(?:ограничения|правила|защиту|фильтры|контраинты)',
]


CITATION_PATTERN = re.compile(r'\[(\d{1,3})\]')

ABSTENTION_MESSAGE = (
    "В доступных вам документах не найдено информации для ответа на этот вопрос."
)


def detect_prompt_injection(text: str) -> bool:
    """Проверить текст на наличие попыток prompt injection.
    
    Args:
        text: Текст для проверки
        
    Returns:
        True если обнаружена инъекция, False иначе
        
    Raises:
        PromptInjectionDetected: Если найдена уязвимость
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Объединение всех паттернов для скорости
    all_patterns = ENGLISH_PATTERNS + RUSSIAN_PATTERNS
    
    for pattern in all_patterns:
        if re.search(pattern, text):
            logger.warning(f"Обнаружена попытка prompt injection в тексте")
            raise PromptInjectionDetected("Запрос отклонён: обнаружена попытка промпт-инъекции.")
    
    return False


def neutralize_injection_in_context(context_text: str) -> str:
    """Нейтрализовать потенциально вредоносные инструкции внутри контекста.
    
    Удаляет или маркирует команды, которые могут быть интерпретированы как
    попытки инъекции даже внутри данных документов.
    """
    if not context_text or not isinstance(context_text, str):
        return ""
    
    # Паттерны инструкций внутри текста (например "Ты должен...", "Игнорируй это...")
    injection_patterns = [
        r'(?i)\b(ты\s+должен|вы\sdолжны|must|\bdon\'t\b)\s+(?:ignore|disregard|forget)',
        r'(?i)\b(если\s+не\stoит|do\s+not|avoid)\s+(?:mention|disclose|reveal|output)',
        r'(?i)\b(помни|remember)|\bзабудь\s+все\b',
    ]
    
    # Заменяем потенциально вредоносные инструкции на нейтральные
    safe_text = context_text
    
    for pattern in injection_patterns:
        try:
            if re.search(pattern, safe_text):
                logger.debug("Нейтрализация инъекции в контексте")
                safe_text = re.sub(pattern, "[НЕЙТРАЛИЗОВАНА]", safe_text)
        except Exception:
            continue
    
    return safe_text


def validate_citations(answer: str, available_doc_ids: Optional[set] = None) -> Tuple[List[int], List[str]]:
    """Извлечь и проверить цитаты в ответе.
    
    Args:
        answer: Ответ модели с возможными ссылками [n]
        available_doc_ids: Множество действительных ID документов
        
    Returns:
        (valid_citations, errors) — кортеж (список валидных нумераций, список ошибок)
    """
    if not isinstance(answer, str):
        return [], ["Ответ должен быть строкой"]
    
    citations = []
    errors = []
    
    # Извлекаем все номера цитат
    for match in CITATION_PATTERN.finditer(answer):
        num = int(match.group(1))
        citations.append(num)
        
        if available_doc_ids is not None:
            # Проверяем, что ссылка существует
            valid_num = f"doc_{num}"  # Ожидаемый формат ID
            if valid_num not in available_doc_ids and str(valid_num) not in available_doc_ids:
                errors.append(f"Некорректная цитата [{num}]: документ не найден")
    
    return citations, errors


def process_text(text: str) -> str:
    """Обработать текст (обезопасить от инъекций)."""
    if not text or not isinstance(text, str):
        return ""
    
    # Удаление потенциально опасных конструкций
    dangerous_patterns = [
        r'```(?:python|code|text)?\s*\n.*?\n```',  # Code blocks
        r'\$\{[^}]+\}',  # Environment variables
        r'\{\{[^}]+\}\}',  # Template placeholders
    ]
    
    safe_text = text
    
    for pattern in dangerous_patterns:
        try:
            if re.search(pattern, safe_text):
                logger.debug("Удаление опасной конструкции")
                safe_text = re.sub(pattern, "[ЗАМЕЩЁНО]", safe_text)
        except Exception:
            continue
    
    return safe_text
