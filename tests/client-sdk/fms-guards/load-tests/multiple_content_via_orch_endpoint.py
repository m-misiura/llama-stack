import asyncio
import time
from statistics import mean, stdev
from datetime import datetime
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms import get_adapter_impl
from llama_stack.providers.remote.safety.fms.config import (
    BaseContentDetectionConfig,
    ContentDetectionConfig,
)


async def run_single_test(
    adapter, content, case_name, expected_violation, expected_detector
):
    start_time = time.time()
    messages = [UserMessage(role="user", content=content)]
    response = await adapter.run_shield("test-shield", messages)
    duration = time.time() - start_time

    violation_detected = response.violation is not None
    assert (
        violation_detected == expected_violation
    ), f"Expected violation: {expected_violation}, got: {violation_detected}"

    if response.violation:
        # Verify guardrails metadata
        assert "detector_id" in response.violation.metadata
        assert response.violation.metadata["detector_id"] == expected_detector
        assert "detection_type" in response.violation.metadata
        assert "score" in response.violation.metadata
        assert "start" in response.violation.metadata
        assert "end" in response.violation.metadata
        assert "text" in response.violation.metadata

    return {
        "case": case_name,
        "duration": duration,
        "violation": violation_detected,
        "detector": expected_detector if violation_detected else None,
    }


async def load_test(num_requests=100, concurrent_requests=10):
    config = ContentDetectionConfig(
        detectors=[
            BaseContentDetectionConfig(
                base_url="https://guardrails-nlp-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
                detector_id="lol",
                confidence_threshold=0.5,
            )
        ],
        use_orchestrator_api=True,
        guardrails_detectors={"hap": {}, "suicide": {}},
    )

    adapter = await get_adapter_impl(config, None)

    class MockShieldStore:
        async def get_shield(self, shield_id):
            return Shield(
                identifier="test-shield",
                provider_id="fms",
                provider_resource_id="hap",
                params={},
            )

    adapter.shield_store = MockShieldStore()

    test_cases = [
        ("Suicidal", "I feel really sad, depressed and suicidal", True, "suicide"),
        ("Harmful", "You dotard, I really hate this stuff", True, "hap"),
        ("Safe", "The weather is nice today", False, None),
    ]

    results = []
    start_time = time.time()
    total_violations = 0
    batch_number = 0

    print(
        f"\nStarting mult-detector routed via the orchestrator load test at {datetime.now()}"
    )
    print(f"Requests: {num_requests}, Concurrent: {concurrent_requests}")
    print("=" * 50)

    for i in range(0, num_requests, concurrent_requests):
        batch_number += 1
        batch_size = min(concurrent_requests, num_requests - i)
        tasks = []

        for _ in range(batch_size):
            case_name, content, expected_violation, expected_detector = test_cases[
                i % len(test_cases)
            ]
            tasks.append(
                run_single_test(
                    adapter, content, case_name, expected_violation, expected_detector
                )
            )

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        batch_violations = sum(1 for r in batch_results if r["violation"])
        total_violations += batch_violations
        print(f"Batch {batch_number}: {batch_size}/{batch_size} successful")

    total_time = time.time() - start_time
    durations = [r["duration"] for r in results]

    print(f"\nLoad Test Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Success rate: {len(results)}/{num_requests}")
    print(f"Violation rate: {total_violations}/{num_requests}")
    print(f"Average response time: {mean(durations):.3f}s")
    print(f"Standard deviation: {stdev(durations):.3f}s")
    print(f"Requests per second: {num_requests/total_time:.2f}")


if __name__ == "__main__":
    asyncio.run(load_test(num_requests=100, concurrent_requests=10))
