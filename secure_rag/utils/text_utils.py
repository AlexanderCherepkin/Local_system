"""Утилиты для обработки текста в RAG системе."""

import os
import re
from typing import List, Counter

class TextUtils:
    TEXT_EXTENSIONS = {'.txt', '.log', '.md', '.rst', '.html', '.htm'}
    
    def read_file(self, file_path: str) -> str:
        """Безопасное чтение файла с перебором кодировок вместо слепого except."""
        encodings = ['utf-8', 'cp1251', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise IOError(f"Ошибка чтения файла {file_path}: {e}")
        raise UnicodeDecodeError(f"Не удалось декодировать файл {file_path} стандартными кодировками.")

    def process_text(self, file_path: str, raw_text: str) -> str:
        _, extension = os.path.splitext(file_path.lower())
        
        # Корректный порядок проверки: специфичные форматы проверяются первыми
        if extension == '.md':
            return self._preprocess_markdown(raw_text)
        elif extension in {'.html', '.htm'}:
            return self._preprocess_html(raw_text)
        elif extension in self.TEXT_EXTENSIONS:
            return raw_text
        else:
            raise ValueError(f"Неподдерживаемое расширение файла: {extension}")

    def _preprocess_markdown(self, text: str) -> str:
        # Логика предобработки Markdown
        return text

    def _preprocess_html(self, text: str) -> str:
        # Логика предобработки HTML
        return text

    def split_text_by_sentences(self, text: str, max_length: int = 500) -> List[str]:
        """Разбиение текста на предложения с соблюдением max_length (':' исключено из конца предложения)."""
        # Разделяем по точкам, восклицательным и вопросительным знакам
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        raw_sentences = sentence_endings.split(text)
        
        sentences = []
        current_chunk = ""
        
        for sentence in raw_sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    sentences.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            sentences.append(current_chunk)
            
        return sentences

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Частотное извлечение ключевых слов с базовым стоп-листом."""
        stop_words = {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'ее', 'мне', 'было', 'вот', 'от'}
        words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]
        
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(top_n)]