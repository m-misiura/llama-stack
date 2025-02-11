import asyncio
import time
from statistics import mean, stdev
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List

from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms import get_adapter_impl
from llama_stack.providers.remote.safety.fms.config import (
    BaseContentDetectionConfig,
    ContentDetectionConfig,
)


@dataclass
class TestCase:
    name: str
    content: str
    expect_detection: bool
    detector_params: Dict[str, List[str]]
    expected_detector: Optional[str] = None


@dataclass
class TestResult:
    duration: float
    success: bool
    detection_found: bool
    detector_id: Optional[str] = None
    error: Optional[str] = None


def generate_detector_configs(num_detectors: int = 6) -> Dict[str, Dict]:
    """Generate detector configurations for multiple regex detectors"""
    return {
        f"regex-{i}": {
            "regex": ["email"]  # Simplified structure matching API expectations
        }
        for i in range(num_detectors)
    }


async def run_single_test(adapter, case: TestCase) -> TestResult:
    start_time = time.time()
    try:
        messages = [UserMessage(role="user", content=case.content)]

        # Match the format from successful curl request
        params = {"detectors": generate_detector_configs(6), "content": case.content}

        response = await adapter.run_shield("test-shield", messages, params)

        detection_found = response.violation is not None
        assert detection_found == case.expect_detection

        if detection_found:
            metadata = response.violation.metadata
            assert all(
                key in metadata for key in ["detector_id", "detection_type", "score"]
            )
            # Verify detector ID matches expected format
            assert metadata["detector_id"].startswith(
                "regex-"
            ), f"Unexpected detector: {metadata['detector_id']}"

        return TestResult(
            duration=time.time() - start_time,
            success=True,
            detection_found=detection_found,
            detector_id=(
                response.violation.metadata["detector_id"] if detection_found else None
            ),
        )
    except Exception as e:
        return TestResult(
            duration=time.time() - start_time,
            success=False,
            detection_found=False,
            error=str(e),
        )


async def load_test(
    num_requests: int = 100, concurrent_requests: int = 1, num_detectors: int = 6
):
    # Generate configurations for all detectors
    detector_configs = generate_detector_configs(num_detectors)

    config = ContentDetectionConfig(
        detectors=[
            BaseContentDetectionConfig(
                base_url="https://guardrails-nlp-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
                detector_id=f"regex-{i}",  # Match the format used in generate_detector_configs
                confidence_threshold=0.5,
            )
            for i in range(num_detectors)
        ],
        use_orchestrator_api=True,
        guardrails_detectors=detector_configs,
    )

    test_cases = [
        TestCase(
            name="Email Detection - Multiple Detectors",
            content="Your email is test@ibm.com! Another email is test2@ibm.com",
            expect_detection=True,
            detector_params={"regex": ["email"]},
            expected_detector=None,
        ),
        TestCase(
            name="Safe Content - Multiple Detectors",
            content="This is a safe message without any PII",
            expect_detection=False,
            detector_params={"regex": ["email"]},
            expected_detector=None,
        ),
    ]

    adapter = await get_adapter_impl(config, None)
    adapter.shield_store = MockShieldStore()

    results: List[TestResult] = []
    total_start = time.time()

    print(f"\nStarting Multi-Detector Orchestrator Test at {datetime.now()}")
    print(f"Requests: {num_requests}, Concurrent: {concurrent_requests}")
    print(f"Number of Detectors: {num_detectors}")
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
    detections = [r for r in successful if r.detection_found]

    print(f"Total time: {total_time:.2f}s")
    print(f"Success rate: {len(successful)}/{len(results)}")
    print(f"Detection rate: {len(detections)}/{len(successful)}")

    if successful:
        durations = [r.duration for r in successful]
        print(f"Average response time: {mean(durations):.3f}s")
        print(f"Standard deviation: {stdev(durations):.3f}s")

    print(f"Requests per second: {total_requests/total_time:.2f}")

    if len(results) != len(successful):
        print("\nErrors:")
        for r in [r for r in results if not r.success]:
            print(f"- {r.error}")

    detector_stats = {}
    for r in successful:
        if r.detector_id:
            if r.detector_id not in detector_stats:
                detector_stats[r.detector_id] = []
            detector_stats[r.detector_id].append(r.duration)

    if detector_stats:
        print("\nDetector Statistics:")
        for detector_id, times in detector_stats.items():
            print(f"\n{detector_id}:")
            print(f"  Count: {len(times)}")
            print(f"  Average time: {mean(times):.3f}s")
            print(f"  Standard deviation: {stdev(times):.3f}s")

    with open("results_orchestrator_multi_regex.csv", "w") as f:
        f.write("request_number,duration,detection_found,detector_id,test_case\n")
        for i, r in enumerate(successful):
            test_case = (
                test_cases[i % len(test_cases)].name
                if "test_cases" in locals()
                else "unknown"
            )
            f.write(
                f"{i+1},{r.duration:.3f},{r.detection_found},{r.detector_id or ''},{test_case}\n"
            )


class MockShieldStore:
    async def get_shield(self, shield_id):
        return Shield(
            identifier="test-shield",
            provider_id="fms",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run multi-detector orchestrator load test"
    )
    parser.add_argument(
        "--num-requests", type=int, default=100, help="Number of requests to send"
    )
    parser.add_argument(
        "--concurrent", type=int, default=1, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--num-detectors", type=int, default=6, help="Number of regex detectors to test"
    )

    args = parser.parse_args()

    asyncio.run(
        load_test(
            num_requests=args.num_requests,
            concurrent_requests=args.concurrent,
            num_detectors=args.num_detectors,
        )
    )
