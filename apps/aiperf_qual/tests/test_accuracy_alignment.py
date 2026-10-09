# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.graders.multiple_choice import MultipleChoiceGrader
from aiperf.common.models import MetricRecordMetadata
from aiperf.common.models.record_models import TextResponseData
from aiperf.dataset.dataset_samplers import SequentialSampler

from trtmc_aiperf_plugins import PTH_LINE

# Exercise the same startup hook used by the AIPerf record-processor subprocess.
exec(PTH_LINE)


def processor():
    value = object.__new__(AccuracyRecordProcessor)
    value.grader = MultipleChoiceGrader(run=None)
    value._grader_name = "multiple_choice"
    value._log_grading_detail = lambda *args: None
    value.on_dataset_configured(SimpleNamespace(conversations=[
        SimpleNamespace(conversation_id="ungraded", accuracy_ground_truth=None, accuracy_task=None),
        *[SimpleNamespace(conversation_id=f"question-{i}", accuracy_ground_truth=answer, accuracy_task=f"task-{i}")
          for i, answer in enumerate("ABCD")],
    ]))
    return value


def grade(value, conversation, session, answer, phase="profiling"):
    metadata = MetricRecordMetadata.model_construct(session_num=session, conversation_id=conversation,
        x_request_id=f"{phase}-{session}", worker_id="worker", benchmark_phase=phase, request_end_ns=1)
    record = SimpleNamespace(content_responses=[SimpleNamespace(data=TextResponseData(text=answer))])
    result = asyncio.run(value.process_record(record, metadata))
    assert metadata.session_num == result.session_num == session
    assert result.conversation_id == conversation and result.x_request_id == metadata.x_request_id
    return result


@pytest.mark.parametrize("warmup", [0, 1, 3, 5])
def test_gold_and_task_follow_sampled_question_across_phase_reset_and_wrap(warmup):
    value = processor()
    sampler = SequentialSampler([f"question-{i}" for i in range(4)])
    for session in range(warmup):
        conversation = sampler.next_conversation_id()
        result = grade(value, conversation, session, "ABCD"[int(conversation[-1])], "warmup")
        assert result.passed
    # Profiling restarts its credit counter, but keeps the sampler's cursor.
    sampled = [(session, sampler.next_conversation_id()) for session in range(8)]
    # Workers can complete out of order.
    for session, conversation in reversed(sampled):
        index = int(conversation[-1])
        result = grade(value, conversation, session, "ABCD"[index])
        assert result.passed and result.expected.strip() == "ABCD"[index]
        assert result.task == f"task-{index}"


def test_unknown_question_fails_instead_of_using_another_questions_gold():
    with pytest.raises((ValueError, RuntimeError), match="conversation"):
        grade(processor(), "missing", 0, "A")
