"""Тесты утилит обработки текста (secure_rag.utils.text_utils)."""

from __future__ import annotations

import pytest

from secure_rag.utils.text_utils import TextUtils


@pytest.fixture()
def utils() -> TextUtils:
    return TextUtils()


class TestReadFile:
    def test_reads_utf8(self, tmp_path, utils) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("Привет, мир!", encoding="utf-8")

        assert utils.read_file(str(path)) == "Привет, мир!"

    def test_reads_cp1251_fallback(self, tmp_path, utils) -> None:
        path = tmp_path / "legacy.txt"
        path.write_bytes("Текст в кодировке".encode("cp1251"))

        assert utils.read_file(str(path)) == "Текст в кодировке"

    def test_missing_file_raises_ioerror(self, tmp_path, utils) -> None:
        with pytest.raises(IOError):
            utils.read_file(str(tmp_path / "нет_такого.txt"))


class TestProcessText:
    def test_markdown_passthrough(self, utils) -> None:
        assert utils.process_text("doc.md", "# Заголовок") == "# Заголовок"

    def test_html_passthrough(self, utils) -> None:
        assert utils.process_text("doc.html", "<p>x</p>") == "<p>x</p>"

    def test_txt_passthrough(self, utils) -> None:
        assert utils.process_text("doc.txt", "обычный текст") == "обычный текст"

    def test_log_supported(self, utils) -> None:
        assert utils.process_text("app.log", "log line") == "log line"

    def test_unsupported_extension_raises(self, utils) -> None:
        with pytest.raises(ValueError):
            utils.process_text("doc.exe", "MZ")


class TestSplitText:
    def test_short_text_single_chunk(self, utils) -> None:
        chunks = utils.split_text_by_sentences("Одно предложение. Второе.", max_length=500)

        assert len(chunks) == 1

    def test_long_text_split(self, utils) -> None:
        text = " ".join(f"Предложение номер {i} довольно длинное." for i in range(20))
        chunks = utils.split_text_by_sentences(text, max_length=100)

        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_empty_text(self, utils) -> None:
        assert utils.split_text_by_sentences("") == []


class TestExtractKeywords:
    def test_stop_words_filtered(self, utils) -> None:
        keywords = utils.extract_keywords("и в на что он все она так его")

        assert keywords == []

    def test_frequent_words_first(self, utils) -> None:
        text = "отпуск отпуск больничный отпуск командировка"

        assert utils.extract_keywords(text, top_n=1) == ["отпуск"]

    def test_top_n_respected(self, utils) -> None:
        keywords = utils.extract_keywords("отпуск больничный командировка", top_n=2)

        assert len(keywords) == 2
