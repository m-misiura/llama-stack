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


async def run_single_test(adapter, content, case_name):
    start_time = time.time()
    messages = [UserMessage(role="user", content=content)]
    response = await adapter.run_shield("test-shield", messages)
    duration = time.time() - start_time
    return {
        "case": case_name,
        "duration": duration,
        "violation": response.violation is not None,
    }


async def load_test(num_requests=100, concurrent_requests=10):
    # Setup config
    config = ContentDetectionConfig(
        detectors=[
            BaseContentDetectionConfig(
                base_url="http://hap-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
                detector_id="hap",
                confidence_threshold=0.8,
            )
        ]
    )

    adapter = await get_adapter_impl(config, None)
    adapter.shield_store = MockShieldStore()

    test_cases = [
        ("Harmful", "You dotard, I really hate this stuff"),
        ("Safe", "The weather is nice today"),
    ]

    results = []
    start_time = time.time()

    print(f"\nStarting load test at {datetime.now()}")
    print(f"Requests: {num_requests}, Concurrent: {concurrent_requests}")
    print("=" * 50)

    for i in range(0, num_requests, concurrent_requests):
        batch_size = min(concurrent_requests, num_requests - i)
        tasks = []

        for _ in range(batch_size):
            case_name, content = test_cases[i % len(test_cases)]
            tasks.append(run_single_test(adapter, content, case_name))

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        print(f"Completed {len(results)}/{num_requests} requests")

    total_time = time.time() - start_time
    durations = [r["duration"] for r in results]

    print("\nLoad Test Results:")
    print(f"Total time: {total_time:.2f}s")
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
