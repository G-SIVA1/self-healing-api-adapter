from __future__ import annotations

from datetime import datetime, timezone

import pytest

from self_healing_agent.agent.history import RepairAttempt
from self_healing_agent.agent.history_retriever import (
    RepairHistoryRetriever,
)
from self_healing_agent.agent.history_store import (
    RepairHistoryStore,
)


def create_attempt(
    *,
    target_file: str = "examples/broken_api_client.py",
    exception_type: str = "AttributeError",
    iteration: int = 0,
    test_passed: bool = True,
) -> RepairAttempt:
    return RepairAttempt(
        iteration=iteration,
        timestamp=datetime.now(timezone.utc),
        target_file=target_file,
        exception_type=exception_type,
        error_message="API failure",
        model="gemini-3.6-flash",
        explanation="Repair explanation",
        confidence=0.95,
        original_code="old code",
        generated_code="new code",
        validation_command=(
            "python",
            "-m",
            "pytest",
            "-q",
        ),
        test_passed=test_passed,
        test_output=(
            "1 passed"
            if test_passed
            else "1 failed"
        ),
        failure_type=None if test_passed else "assertion",
        failure_summary=(
            None
            if test_passed
            else "Validation failed"
        ),
        failure_details=(
            None
            if test_passed
            else "Test failed"
        ),
        status=(
            "success"
            if test_passed
            else "failed"
        ),
    )


def create_store(tmp_path) -> RepairHistoryStore:
    return RepairHistoryStore(
        tmp_path / "history.db"
    )


def test_find_by_target_file(tmp_path) -> None:
    store = create_store(tmp_path)

    store.save(
        create_attempt(
            target_file="examples/client_a.py",
            iteration=0,
        )
    )

    store.save(
        create_attempt(
            target_file="examples/client_b.py",
            iteration=1,
        )
    )

    store.save(
        create_attempt(
            target_file="examples/client_a.py",
            iteration=2,
        )
    )

    retriever = RepairHistoryRetriever(store)

    results = retriever.find_by_target_file(
        "examples/client_a.py"
    )

    assert len(results) == 2

    assert (
        results[0].attempt.iteration
        == 2
    )

    assert (
        results[1].attempt.iteration
        == 0
    )

    assert (
        results[0].relevance_reason
        == "Previous repair targeted the same file."
    )


def test_find_by_target_file_respects_limit(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    for iteration in range(5):
        store.save(
            create_attempt(
                iteration=iteration
            )
        )

    retriever = RepairHistoryRetriever(store)

    results = retriever.find_by_target_file(
        "examples/broken_api_client.py",
        limit=2,
    )

    assert len(results) == 2
    assert results[0].attempt.iteration == 4
    assert results[1].attempt.iteration == 3


def test_find_by_exception(tmp_path) -> None:
    store = create_store(tmp_path)

    store.save(
        create_attempt(
            exception_type="AttributeError",
            iteration=0,
        )
    )

    store.save(
        create_attempt(
            exception_type="TypeError",
            iteration=1,
        )
    )

    store.save(
        create_attempt(
            exception_type="AttributeError",
            iteration=2,
        )
    )

    retriever = RepairHistoryRetriever(store)

    results = retriever.find_by_exception(
        "AttributeError"
    )

    assert len(results) == 2
    assert results[0].attempt.iteration == 2
    assert results[1].attempt.iteration == 0

    assert (
        results[0].relevance_reason
        == (
            "Previous repair encountered the "
            "same exception type."
        )
    )


def test_find_similar_matches_file_and_exception(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    store.save(
        create_attempt(
            target_file="examples/client_a.py",
            exception_type="AttributeError",
            iteration=0,
        )
    )

    store.save(
        create_attempt(
            target_file="examples/client_a.py",
            exception_type="TypeError",
            iteration=1,
        )
    )

    store.save(
        create_attempt(
            target_file="examples/client_b.py",
            exception_type="AttributeError",
            iteration=2,
        )
    )

    store.save(
        create_attempt(
            target_file="examples/client_a.py",
            exception_type="AttributeError",
            iteration=3,
        )
    )

    retriever = RepairHistoryRetriever(store)

    results = retriever.find_similar(
        target_file="examples/client_a.py",
        exception_type="AttributeError",
    )

    assert len(results) == 2
    assert results[0].attempt.iteration == 3
    assert results[1].attempt.iteration == 0

    assert (
        results[0].relevance_reason
        == (
            "Previous repair matches both the "
            "target file and exception type."
        )
    )


def test_find_successful_returns_only_successes(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    store.save(
        create_attempt(
            iteration=0,
            test_passed=False,
        )
    )

    store.save(
        create_attempt(
            iteration=1,
            test_passed=True,
        )
    )

    store.save(
        create_attempt(
            iteration=2,
            test_passed=False,
        )
    )

    store.save(
        create_attempt(
            iteration=3,
            test_passed=True,
        )
    )

    retriever = RepairHistoryRetriever(store)

    results = retriever.find_successful(
        target_file="examples/broken_api_client.py",
        exception_type="AttributeError",
    )

    assert len(results) == 2
    assert results[0].attempt.iteration == 3
    assert results[1].attempt.iteration == 1

    assert all(
        result.attempt.test_passed
        for result in results
    )


def test_no_matching_target_file_returns_empty(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    store.save(create_attempt())

    retriever = RepairHistoryRetriever(store)

    results = retriever.find_by_target_file(
        "examples/other.py"
    )

    assert results == []


def test_no_matching_exception_returns_empty(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    store.save(create_attempt())

    retriever = RepairHistoryRetriever(store)

    results = retriever.find_by_exception(
        "TypeError"
    )

    assert results == []


def test_no_matching_similar_repairs_returns_empty(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    store.save(create_attempt())

    retriever = RepairHistoryRetriever(store)

    results = retriever.find_similar(
        target_file="examples/other.py",
        exception_type="TypeError",
    )

    assert results == []


def test_empty_target_file_is_rejected(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    retriever = RepairHistoryRetriever(store)

    with pytest.raises(
        ValueError,
        match="target_file cannot be empty",
    ):
        retriever.find_by_target_file("")


def test_empty_exception_type_is_rejected(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    retriever = RepairHistoryRetriever(store)

    with pytest.raises(
        ValueError,
        match="exception_type cannot be empty",
    ):
        retriever.find_by_exception("")


def test_invalid_limit_is_rejected(
    tmp_path,
) -> None:
    store = create_store(tmp_path)

    retriever = RepairHistoryRetriever(store)

    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        retriever.find_by_target_file(
            "examples/client.py",
            limit=0,
        )