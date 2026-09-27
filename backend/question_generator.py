"""Question generation via AGNES LLM (Qwen3.8-Flash-Next).

Takes a context block (OCR text from one or more pages) and generates
structured multiple-choice questions grounded in the source text.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import VLM_API_KEY, VLM_BASE_URL, VLM_MODEL

log = logging.getLogger("qgen")

SYSTEM_PROMPT = """你是一个出题助手。基于用户提供的书籍原文片段，生成高质量的选择题。

要求：
1. 每题必须能在原文中找到明确依据
2. 题目要考察理解，不是简单摘抄
3. 错误选项要具有迷惑性但不能错得离谱
4. 答案唯一
5. 输出必须是合法 JSON

输出 JSON 格式：
{
  "questions": [
    {
      "id": 1,
      "type": "single_choice",
      "difficulty": "easy" | "medium" | "hard",
      "question": "题干",
      "options": ["选项A", "选项B", "选项C", "选项D"],
      "answer": "A",
      "explanation": "答案解析，要引用原文依据",
      "source_snippet": "原文中能支持该题的关键片段（30字以内）"
    }
  ]
}
"""


def _make_client() -> OpenAI:
    return OpenAI(api_key=VLM_API_KEY, base_url=VLM_BASE_URL)


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if m:
        return m.group(1)
    return t


def _try_repair_json(text: str) -> str | None:
    """Attempt to fix truncated JSON by closing open brackets/quotes.

    Returns repaired JSON string or None.
    """
    t = text.rstrip().rstrip(",")
    # Count open brackets
    stack = []
    in_str = False
    esc = False
    for ch in t:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                stack.append("]")
            elif ch == "{":
                stack.append("}")
            elif ch in "]}":
                if stack and stack[-1] == ch:
                    stack.pop()

    # If we're inside a string, close it
    if in_str:
        t = t + '"'
    # Close any open brackets in reverse order
    while stack:
        t = t + stack.pop()
    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        return None


def generate_questions(
    context_text: str,
    num_questions: int = 5,
    difficulty: str = "mixed",
    book_id: str = "",
    page_range: str = "",
) -> dict[str, Any]:
    """Generate questions from a context block.

    Returns:
        {"questions": [...], "raw_response": "..."}

    Raises:
        ValueError: if LLM returned unparseable JSON or empty questions.
    """
    if not context_text or not context_text.strip():
        raise ValueError("Context text is empty")

    user_prompt = (
        f"书籍ID: {book_id}\n"
        f"页码范围: {page_range}\n"
        f"难度: {difficulty}\n"
        f"题目数量: {num_questions}\n\n"
        '书籍原文片段：\n"""\n'
        f"{context_text}\n"
        '"""\n\n'
        f"请基于以上原文，生成 {num_questions} 道选择题。"
    )

    client = _make_client()
    log.info(f"Generating {num_questions} questions, context {len(context_text)} chars")
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=4000,
    )
    raw = resp.choices[0].message.content or ""
    log.info(f"LLM response: {len(raw)} chars, finish={resp.choices[0].finish_reason}")

    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse failed: {e}, attempting repair")
        repaired = _try_repair_json(cleaned)
        if repaired:
            try:
                parsed = json.loads(repaired)
                log.info("JSON repaired successfully")
            except json.JSONDecodeError as e2:
                log.error(f"Repair failed: {e2}")
                raise ValueError(f"LLM returned invalid JSON (repair failed): {e}")
        else:
            log.error(f"JSON repair not possible. Raw head: {raw[:400]}")
            raise ValueError(f"LLM returned invalid JSON: {e}")

    questions = parsed.get("questions", [])
    # Validate structure
    valid = []
    for q in questions:
        try:
            opts = q.get("options", [])
            ans = q.get("answer", "")
            if not q.get("question") or len(opts) < 2 or ans not in [chr(ord('A')+i) for i in range(len(opts))]:
                continue
            valid.append({
                "id": q.get("id", len(valid)+1),
                "type": q.get("type", "single_choice"),
                "difficulty": q.get("difficulty", "medium"),
                "question": q["question"],
                "options": opts,
                "answer": ans,
                "explanation": q.get("explanation", ""),
                "source_snippet": q.get("source_snippet", ""),
            })
        except Exception as e:
            log.warning(f"Skipping malformed question: {e}")

    return {"questions": valid, "raw_response": raw}


def get_all_context_for_book(kb, book_id: str) -> str:
    """Concatenate all pages of a book in order."""
    pages = sorted([c for c in kb.chunks if c.book_id == book_id], key=lambda x: x.page_no)
    parts = []
    for c in pages:
        parts.append(f"【第{c.page_no}页】\n{c.text}")
    return "\n\n".join(parts)
