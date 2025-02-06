import asyncio
import time
from datetime import datetime
from statistics import mean, stdev
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms import get_adapter_impl
from llama_stack.providers.remote.safety.fms.config import (
    DetectorConfig,
    FMSModelConfig,
)


async def run_single_test(adapter, content, expected_violation):
    start_time = time.time()
    try:
        messages = [UserMessage(role="user", content=content)]
        response = await adapter.run_shield("test-shield", messages)

        violation_detected = response.violation is not None
        assert (
            violation_detected == expected_violation
        ), f"Expected violation: {expected_violation}, got: {violation_detected}"

        if response.violation:
            metadata = response.violation.metadata
            assert all(
                key in metadata for key in ["detector_id", "start", "end", "text"]
            )

        return {
            "duration": time.time() - start_time,
            "success": True,
            "violation": violation_detected,
        }
    except Exception as e:
        return {"duration": time.time() - start_time, "success": False, "error": str(e)}


async def load_test(num_requests=100, concurrent_requests=10):
    config = FMSModelConfig(
        detectors=[
            DetectorConfig(
                base_url="https://guardrails-nlp-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
                detector_id="hap",
                confidence_threshold=0.8,
            )
        ],
        use_orchestrator_api=True,
    )

    adapter = await get_adapter_impl(config, None)
    adapter.shield_store = MockShieldStore()

    test_cases = [
        ("Harmful", "You dotard, I really hate this stuff", True),
        ("Safe", "The weather is nice today", False),
    ]

    results = []
    total_start = time.time()

    print(f"\nStarting FMS Guardrails Load Test at {datetime.now()}")
    print(f"Total requests: {num_requests}, Concurrent: {concurrent_requests}")
    print("=" * 50)

    for i in range(0, num_requests, concurrent_requests):
        batch_size = min(concurrent_requests, num_requests - i)
        tasks = []

        for _ in range(batch_size):
            case_name, content, expected = test_cases[i % len(test_cases)]
            tasks.append(run_single_test(adapter, content, expected))

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        successful = len([r for r in batch_results if r["success"]])
        print(
            f"Batch {i//concurrent_requests + 1}: {successful}/{batch_size} successful"
        )

    total_time = time.time() - total_start
    successful_results = [r for r in results if r["success"]]
    durations = [r["duration"] for r in successful_results]

    print("\nLoad Test Summary:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Success rate: {len(successful_results)}/{len(results)}")
    if successful_results:
        print(f"Average response time: {mean(durations):.3f}s")
        print(f"Standard deviation: {stdev(durations):.3f}s")
    print(f"Requests per second: {num_requests/total_time:.2f}")


class MockShieldStore:
    async def get_shield(self, shield_id):
        return Shield(
            identifier="test-shield",
            provider_id="fms",
        )


if __name__ == "__main__":
    asyncio.run(load_test(num_requests=100, concurrent_requests=10))
