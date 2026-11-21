"""PersonaMem-v2 4-way MCQ prompts, scoring, and end-to-end runner wiring."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from trisynapse_memory.benchmarks import runner
from trisynapse_memory.benchmarks.mcq import (
    labeled_mcq_options,
    mcq_answer_question,
    parse_mcq_choice,
    score_mcq_prediction,
)
from trisynapse_memory.adapters.benchmarks import get_adapter
from trisynapse_memory.adapters.benchmarks.base import BenchmarkQuestion
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.prompts import load_prompt


class DeterministicEmbedder:
    model_name = "mcq-test-embedding-v1"
    provider_name = "test"

    def encode(self, texts: list[str]) -> list[list[float]]:
        values = []
        for text in texts:
            vector = [0.0] * 32
            for token in text.lower().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1
            norm = sum(item * item for item in vector) ** 0.5 or 1
            values.append([item / norm for item in vector])
        return values


def test_parse_mcq_choice_accepts_letter_or_exact_option() -> None:
    labeled = [("A", "Green tea"), ("B", "Coffee"), ("C", "Water"), ("D", "Juice")]
    assert parse_mcq_choice("B", labeled) == "B"
    assert parse_mcq_choice("b.", labeled) == "B"
    assert parse_mcq_choice("Coffee", labeled) == "B"
    assert parse_mcq_choice("I am not sure", labeled) is None


def test_mcq_option_order_is_stable_and_not_always_gold_first() -> None:
    question = BenchmarkQuestion(
        id="p1:q0",
        question="What tea do I like?",
        gold="Green tea in the morning.",
        choices=(
            "Black coffee in the evening.",
            "A double espresso before bed.",
            "Only water, never tea.",
        ),
    )
    labeled, correct = labeled_mcq_options(question)
    again, correct_again = labeled_mcq_options(question)
    assert labeled == again
    assert correct == correct_again
    assert correct in "ABCD"
    assert {text for _, text in labeled} == {question.gold, *question.choices}
    prompt = mcq_answer_question(question)
    assert prompt.startswith("What tea do I like?")
    assert "Choose the best personalized response" in prompt
    assert all(text in prompt for _, text in labeled)


def test_personamem_loads_incorrect_answers_as_choices(tmp_path: Path) -> None:
    csv_path = _write_personamem_csv(tmp_path)
    case = next(iter(get_adapter("personamem").load_cases(csv_path)))
    question = case.questions[0]
    assert question.gold == "Green tea in the morning."
    assert question.choices == (
        "Black coffee in the evening.",
        "A double espresso before bed.",
        "Only water, never tea.",
    )


def test_personamem_end_to_end_scores_mcq_without_judge(tmp_path: Path, monkeypatch) -> None:
    csv_path = _write_personamem_csv(tmp_path)
    question = next(iter(get_adapter("personamem").load_cases(csv_path))).questions[0]
    _, correct_letter = labeled_mcq_options(question)
    completion = _McqCompletion(correct_letter)
    seen: list[tuple[str, str | None]] = []

    def open_engine(store_path, mode, configuration=None):
        del mode, configuration
        engine = MemoryEngine.open(
            store_path,
            embedder=DeterministicEmbedder(),
            completion=completion,
            auto_process=False,
        )
        original = engine.query

        def query(prompt, answer_question=None, **kwargs):
            result = original(prompt, answer_question=answer_question, **kwargs)
            seen.append((prompt, answer_question))
            return result.model_copy(update={"answer": correct_letter, "abstain": False})

        engine.query = query
        return engine

    monkeypatch.setattr(runner, "_open_benchmark_engine", open_engine)
    summary = runner.run_benchmark(
        "personamem",
        csv_path,
        max_questions=1,
        output_root=tmp_path / "runs",
        mode="end-to-end",
        stage="auto",
    )
    artifact = json.loads(
        (tmp_path / "runs" / f"trace_recall_personamem_{summary['run_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    row = artifact["results"][0]
    assert seen
    assert seen[0][0] == "What tea do I like?"
    assert seen[0][1] is not None
    assert "Choose the best personalized response" in seen[0][1]
    assert "Black coffee in the evening." in seen[0][1]
    assert completion.judge_calls == 0
    input_step = next(step for step in row["query_steps"] if step["phase"] == "input")
    assert input_step["output"]["query"] == "What tea do I like?"
    assert row["scoring"] == "mcq"
    assert row["mcq_choice"] == correct_letter
    assert row["accuracy"] == 1.0
    assert "judge" not in row
    assert summary["accuracy"] == 1.0


def test_score_mcq_prediction_is_wrong_on_other_letter() -> None:
    question = BenchmarkQuestion(
        id="p1:q0",
        question="What tea do I like?",
        gold="Green tea in the morning.",
        choices=("Coffee", "Water", "Juice"),
    )
    labeled, correct = labeled_mcq_options(question)
    wrong = next(letter for letter, _ in labeled if letter != correct)
    scored = score_mcq_prediction(wrong, question)
    assert scored["accuracy"] == 0.0
    assert scored["mcq_choice"] == wrong


class _McqCompletion:
    model = "mcq-test"
    provider_name = "test"

    def __init__(self, letter: str) -> None:
        self.letter = letter
        self.answer_users: list[str] = []
        self.judge_calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        if system == load_prompt("benchmark_judge").text:
            self.judge_calls += 1
            raise AssertionError("PersonaMem MCQ must not use the generic judge")
        if system == load_prompt("facts").text:
            return {
                "facts": [{
                    "text": "The user prefers green tea in the morning.",
                    "subject": "user",
                    "relation": "prefers",
                    "object": "green tea",
                    "confidence": 0.9,
                }]
            }
        if system == load_prompt("group_summary").text:
            return {
                "concept_or_topic": "tea",
                "summary": "The user prefers green tea in the morning.",
                "alt_phrasings": [],
            }
        if system == load_prompt("answer").text:
            self.answer_users.append(user)
            return {"answer": self.letter, "abstain": False, "citation_ids": []}
        raise AssertionError(f"unexpected prompt: {system[:80]}")

    __call__ = complete_json


def _write_personamem_csv(tmp_path: Path) -> Path:
    root = tmp_path / "personamem"
    history_dir = root / "data" / "chat_history_32k"
    csv_dir = root / "benchmark" / "text"
    history_dir.mkdir(parents=True)
    csv_dir.mkdir(parents=True)
    (history_dir / "h1.json").write_text(
        json.dumps([
            {"role": "user", "content": "I prefer green tea in the morning."},
            {"role": "assistant", "content": "I will remember that."},
        ]),
        encoding="utf-8",
    )
    csv_path = csv_dir / "benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "persona_id",
                "user_query",
                "correct_answer",
                "incorrect_answers",
                "chat_history_32k_link",
                "related_conversation_snippet",
                "topic_query",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "persona_id": "p1",
            "user_query": "What tea do I like?",
            "correct_answer": "Green tea in the morning.",
            "incorrect_answers": json.dumps([
                "Black coffee in the evening.",
                "A double espresso before bed.",
                "Only water, never tea.",
            ]),
            "chat_history_32k_link": "data/chat_history_32k/h1.json",
            "related_conversation_snippet": "I prefer green tea in the morning.",
            "topic_query": "Health",
        })
    return csv_path
