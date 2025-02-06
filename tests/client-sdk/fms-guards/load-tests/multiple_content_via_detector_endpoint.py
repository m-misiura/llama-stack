import asyncio
import time
from datetime import datetime
from statistics import mean, stdev
from dataclasses import dataclass
from typing import Optional, Dict, List

from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms import get_adapter_impl
from llama_stack.providers.remote.safety.fms.config import (
    DetectorConfig,
    FMSModelConfig,
)


@dataclass
class TestCase:
    name: str
    content: str
    expect_violation: bool
    expected_detector: Optional[str]


@dataclass
class TestResult:
    duration: float
    success: bool
    violation_detected: bool
    detector_id: Optional[str] = None
    error: Optional[str] = None


async def run_single_test(adapter, case: TestCase) -> TestResult:
    start_time = time.time()
    try:
        messages = [UserMessage(role="user", content=case.content)]
        response = await adapter.run_shield("test-shield", messages)

        violation_detected = response.violation is not None
        assert violation_detected == case.expect_violation

        if violation_detected:
            metadata = response.violation.metadata
            assert all(
                key in metadata for key in ["detector_id", "detection_type", "score"]
            )
            if case.expected_detector:
                assert metadata["detector_id"] == case.expected_detector

        return TestResult(
            duration=time.time() - start_time,
            success=True,
            violation_detected=violation_detected,
            detector_id=(
                response.violation.metadata["detector_id"]
                if violation_detected
                else None
            ),
        )
    except Exception as e:
        return TestResult(
            duration=time.time() - start_time,
            success=False,
            violation_detected=False,
            error=str(e),
        )


async def load_test(num_requests: int = 100, concurrent_requests: int = 10):
    config = FMSModelConfig(
        detectors=[
            DetectorConfig(
                base_url="http://hap-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
                detector_id="hap",
                confidence_threshold=0.5,
            ),
            DetectorConfig(
                base_url="http://suicide-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
                detector_id="suicide",
                confidence_threshold=0.5,
            ),
        ]
    )

    test_cases = [
        TestCase(
            "Suicidal", "I feel really sad, depressed and suicidal", True, "suicide"
        ),
        TestCase("Harmful", "You dotard, I really hate this stuff", True, "hap"),
        TestCase("Safe", "The weather is nice today", False, None),
    ]

    adapter = await get_adapter_impl(config, None)
    adapter.shield_store = MockShieldStore()

    results: List[TestResult] = []
    total_start = time.time()

    print(f"\nStarting Multi-Detector Load Test at {datetime.now()}")
    print(f"Requests: {num_requests}, Concurrent: {concurrent_requests}")
    print("=" * 50)

    for i in range(0, num_requests, concurrent_requests):
        batch_size = min(concurrent_requests, num_requests - i)
        tasks = []

        for _ in range(batch_size):
            case = test_cases[i % len(test_cases)]
            tasks.append(run_single_test(adapter, case))

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        successful = len([r for r in batch_results if r.success])
        print(
            f"Batch {i//concurrent_requests + 1}: {successful}/{batch_size} successful"
        )

    total_time = time.time() - total_start

    print("\nLoad Test Results:")
    print_test_summary(results, total_time, num_requests)


def print_test_summary(
    results: List[TestResult], total_time: float, total_requests: int
):
    successful = [r for r in results if r.success]
    violations = [r for r in successful if r.violation_detected]

    print(f"Total time: {total_time:.2f}s")
    print(f"Success rate: {len(successful)}/{len(results)}")
    print(f"Violation rate: {len(violations)}/{len(successful)}")

    if successful:
        durations = [r.duration for r in successful]
        print(f"Average response time: {mean(durations):.3f}s")
        print(f"Standard deviation: {stdev(durations):.3f}s")

    print(f"Requests per second: {total_requests/total_time:.2f}")

    if len(results) != len(successful):
        print("\nErrors:")
        for r in [r for r in results if not r.success]:
            print(f"- {r.error}")


class MockShieldStore:
    async def get_shield(self, shield_id):
        return Shield(
            identifier="test-shield",
            provider_id="fms",
        )


if __name__ == "__main__":
    asyncio.run(load_test(num_requests=100, concurrent_requests=10))
