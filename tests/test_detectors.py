"""Tests for refusal/abstention detection."""
from __future__ import annotations

from ddmi.evaluation.detectors import (
    abstention_rate,
    classify_responses,
    is_abstention,
    is_refusal,
    refusal_rate,
)


class TestRefusalDetection:
    def test_clear_refusal(self) -> None:
        assert is_refusal("I can't help with that request.")

    def test_not_refusal(self) -> None:
        assert not is_refusal("Here is how you can do that:")

    def test_apologetic_refusal(self) -> None:
        assert is_refusal("I'm sorry, but I cannot assist with this.")

    def test_refusal_rate(self) -> None:
        responses = [
            "I can't help with that.",
            "Here you go!",
            "I apologize, but I cannot provide that.",
        ]
        assert abs(refusal_rate(responses) - 2 / 3) < 1e-10


class TestAbstentionDetection:
    def test_clear_abstention(self) -> None:
        assert is_abstention("I don't know the answer to that question.")

    def test_not_abstention(self) -> None:
        assert not is_abstention("The answer is 42.")

    def test_passage_abstention(self) -> None:
        assert is_abstention("The passage does not contain enough information to answer.")

    def test_abstention_rate(self) -> None:
        responses = [
            "I don't know.",
            "Paris is the capital of France.",
            "Not enough information to determine the answer.",
        ]
        assert abs(abstention_rate(responses) - 2 / 3) < 1e-10


class TestClassification:
    def test_priority_refusal_over_abstention(self) -> None:
        # A response that matches both — refusal should win
        text = "I'm sorry, but I don't know and I can't help with that."
        classes = classify_responses([text])
        assert classes[0] == "refusal"

    def test_plain_answer(self) -> None:
        classes = classify_responses(["The capital of France is Paris."])
        assert classes[0] == "answer"
