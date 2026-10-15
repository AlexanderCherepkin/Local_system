import re
from typing import Dict, List, Tuple

# Регулярные выражения для детекции
PATTERNS = {
    "PASSWORD": r"(?i)(?:password|passwd|secret|api_key|token)\s*[:=]\s*([^\s,;]+)",
    "AWS_KEY": r"\bAKIA[0-9A-Z]{16}\b",
    "GITHUB_TOKEN": r"\bgh[pousr]_[a-zA-Z0-9]{36}\b",
    "JWT": r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b",
    "PRIVATE_KEY": r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH|PRIVATE)\s+PRIVATE\s+KEY-----",
    "POSTGRES_URI": r"postgresql://[^:\s]+:[^@\s]+@[^\s/]+/[^\s?]+",
    "PHONE_RU": r"(?:\+7|8)\s*\(?9\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}",
    "SNILS": r"\b\d{3}-\d{3}-\d{3}\s+\d{2}\b",
    "INN": r"\b(?:\d{10}|\d{12})\b",
    "PASSPORT_RU": r"\b\d{2}\s*\d{2}\s*\d{6}\b",
    "OGRN": r"\b(?:1\d{12}|3\d{14})\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,19}\b",
    "PROMPT_INJECTION": r"(?i)(ignore previous instructions|system prompt|disregard all prior)"
}

CRITICAL_TYPES = {"PASSWORD", "AWS_KEY", "GITHUB_TOKEN", "JWT", "PRIVATE_KEY", "POSTGRES_URI"}

def _luhn_check(card_number: str) -> bool:
    digits = [int(c) for c in card_number if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

def process_text(text: str, mode: str = "redact") -> Tuple[str, List[Dict], str]:
    """
    Режимы: 'detect' или 'redact'.
    Возвращает: (processed_text, redactions_applied, max_severity)
    """
    redactions_applied = []
    max_severity = "LOW"
    
    processed_text = text
    # Сортируем паттерны или применяем последовательно
    for p_type, pattern in PATTERNS.items():
        for match in re.finditer(pattern, processed_text):
            val = match.group(0)
            
            # Дополнительная проверка для банковских карт по алгоритму Луна
            if p_type == "CREDIT_CARD" and not _luhn_check(val):
                continue
                
            severity = "CRITICAL" if p_type in CRITICAL_TYPES else "MEDIUM"
            if severity == "CRITICAL":
                max_severity = "CRITICAL"
                
            redactions_applied.append({
                "type": p_type,
                "severity": severity,
                "start": match.start(),
                "end": match.end(),
                "snippet": val[:4] + "***" if len(val) > 4 else "***"
            })

            if mode == "redact":
                replacement = f"[REDACTED:{p_type}]"
                processed_text = processed_text.replace(val, replacement)

    return processed_text, redactions_applied, max_severity