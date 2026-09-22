from __future__ import annotations

import json

import pytest

from self_healing_agent.agent.response_parser import (
    RepairResponseError,
    RepairResponseParser,
)


def create_parser() -> RepairResponseParser:
    return RepairResponseParser()


def create_original_code() -> str:
    return """\
import stripe


def create_customer():
    return stripe.Customer.create(
        email="user@example.com"
    )
"""


def create_valid_response() -> str:
    return json.dumps(
        {
            "explanation": (
                "Replace the deprecated Stripe "
                "Customer API."
            ),
            "confidence": 0.95,
            "replacement_code": """\
import stripe


def create_customer():
    return stripe.customers.create(
        email="user@example.com"
    )
""",
        }
    )


def test_parser_accepts_valid_response() -> None:
    parser = create_parser()

    result = parser.parse(
        response=create_valid_response(),
        target_file="api_client.py",
        original_code=create_original_code(),
    )

    assert result.confidence == 0.95

    assert (
        "deprecated Stripe"
        in result.explanation
    )

    assert (
        "stripe.customers.create"
        in result.patch.replacement_code
    )

    assert (
        result.patch.file_path
        == "api_client.py"
    )


def test_parser_rejects_invalid_json() -> None:
    parser = create_parser()

    with pytest.raises(
        RepairResponseError,
        match="not valid JSON",
    ):
        parser.parse(
            response="this is not json",
            target_file="api_client.py",
            original_code=create_original_code(),
        )


def test_parser_rejects_missing_explanation() -> None:
    parser = create_parser()

    response = json.dumps(
        {
            "confidence": 0.9,
            "replacement_code": "new code",
        }
    )

    with pytest.raises(
        RepairResponseError,
        match="'explanation' must be a string",
    ):
        parser.parse(
            response=response,
            target_file="api_client.py",
            original_code=create_original_code(),
        )


def test_parser_rejects_invalid_confidence() -> None:
    parser = create_parser()

    response = json.dumps(
        {
            "explanation": "Repair",
            "confidence": 1.5,
            "replacement_code": "new code",
        }
    )

    with pytest.raises(
        RepairResponseError,
        match="between 0 and 1",
    ):
        parser.parse(
            response=response,
            target_file="api_client.py",
            original_code=create_original_code(),
        )


def test_parser_rejects_empty_replacement() -> None:
    parser = create_parser()

    response = json.dumps(
        {
            "explanation": "Repair",
            "confidence": 0.9,
            "replacement_code": "",
        }
    )

    with pytest.raises(
        RepairResponseError,
        match="'replacement_code' cannot be empty",
    ):
        parser.parse(
            response=response,
            target_file="api_client.py",
            original_code=create_original_code(),
        )


def test_parser_rejects_unchanged_source() -> None:
    parser = create_parser()

    response = json.dumps(
        {
            "explanation": "Repair",
            "confidence": 0.9,
            "replacement_code": (
                create_original_code()
            ),
        }
    )

    with pytest.raises(
        RepairResponseError,
        match="does not change",
    ):
        parser.parse(
            response=response,
            target_file="api_client.py",
            original_code=create_original_code(),
        )


def test_parser_rejects_confidence_above_one() -> None:
    parser = create_parser()

    response = json.dumps(
        {
            "explanation": "Repair",
            "confidence": 2.0,
            "replacement_code": "new code",
        }
    )

    with pytest.raises(
        RepairResponseError,
        match="between 0 and 1",
    ):
        parser.parse(
            response=response,
            target_file="api_client.py",
            original_code=create_original_code(),
        )