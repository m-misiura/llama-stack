import asyncio
from typing import Optional
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms_refactor import get_adapter_impl
from llama_stack.providers.remote.safety.fms_refactor.config import (
    ContentDetectorConfig,
    BaseDetectorConfig,
)


async def test_content_detector_live():
    """Test content detector with multiple configurations"""

    # Configure main detector with orchestrator API
    main_config = ContentDetectorConfig(
        base_url="https://guardrails-nlp-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
        detector_id="hap",
        confidence_threshold=0.5,
        detector_params={},
        use_orchestrator_api=True,  # Enable orchestrator API
        guardrails_detectors={"hap": {}, "suicide": {}},
    )

    adapter = None
    try:
        adapter = await get_adapter_impl(main_config)

        class MockShieldStore:
            async def get_shield(self, shield_id: str) -> Shield:
                return Shield(
                    identifier="test-shield",
                    provider_id="content",
                    provider_resource_id="hap",
                    params={},
                )

        adapter.shield_store = MockShieldStore()

        test_cases = [
            {
                "name": "Suicidal",
                "content": "I feel really sad, depressed and suicidal",
                "expected_violation": True,
                "expected_detector": "suicide",
            },
            {
                "name": "Harmful",
                "content": "You dotard, I really hate this stuff",
                "expected_violation": True,
                "expected_detector": "hap",
            },
            {
                "name": "Safe",
                "content": "The weather is nice today",
                "expected_violation": False,
                "expected_detector": None,
            },
        ]

        print("\nRunning Content Detector Tests")
        print("============================")

        for case in test_cases:
            print(f"\nTest Case: {case['name']}")
            print(f"Input: {case['content']}")

            messages = [UserMessage(role="user", content=case["content"])]
            response = await adapter.run_shield("test-shield", messages)

            violation_detected = response.violation is not None
            print(f"Violation detected: {violation_detected}")

            assert (
                violation_detected == case["expected_violation"]
            ), f"Expected violation: {case['expected_violation']}, got: {violation_detected}"

            if response.violation:
                print("Violation details:")
                print(f"- Message: {response.violation.user_message}")
                print(f"- Level: {response.violation.violation_level}")
                print(f"- Metadata: {response.violation.metadata}")

                metadata = response.violation.metadata

                # Verify detector identification
                assert "detector_id" in metadata, "Missing detector_id in metadata"
                if case["expected_detector"]:
                    assert (
                        metadata["detector_id"] == case["expected_detector"]
                    ), f"Expected detector {case['expected_detector']}, got {metadata['detector_id']}"

                # Verify required metadata fields
                required_fields = ["detection_type", "score", "text", "start", "end"]
                for field in required_fields:
                    assert field in metadata, f"Missing {field} in metadata"

                # Verify confidence threshold
                assert (
                    float(metadata["score"]) > 0.5
                ), f"Score ({metadata['score']}) should be above confidence threshold (0.5)"

            print("-------------------------------------")

    except Exception as e:
        print(f"Test error: {str(e)}")
        raise
    finally:
        if adapter:
            await adapter.shutdown()


if __name__ == "__main__":
    asyncio.run(test_content_detector_live())
