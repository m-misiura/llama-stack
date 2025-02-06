import asyncio
import os
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms import get_adapter_impl
from llama_stack.providers.remote.safety.fms.config import (
    DetectorConfig,
    FMSModelConfig,
)


async def test_fms_detectors_live():
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
        ],
    )

    adapter = await get_adapter_impl(config, None)

    class MockShieldStore:
        async def get_shield(self, shield_id):
            return Shield(
                identifier="test-shield",
                provider_id="fms",
            )

    adapter.shield_store = MockShieldStore()

    test_cases = [
        ("Suicidal", "I feel really sad, depressed and suicidal", True, "suicide"),
        ("Harmful", "You dotard, I really hate this stuff", True, "hap"),
        ("Safe", "The weather is nice today", False, None),
    ]

    print("\nRunning FMS Multi-Detector Tests...")
    print("===================================")

    for case_name, content, expected_violation, expected_detector in test_cases:
        print(f"\nTest Case: {case_name}")
        print(f"Input: {content}")

        messages = [UserMessage(role="user", content=content)]
        response = await adapter.run_shield("test-shield", messages)

        violation_detected = response.violation is not None
        print(f"Violation detected: {violation_detected}")

        assert (
            violation_detected == expected_violation
        ), f"Expected violation: {expected_violation}, got: {violation_detected}"

        if response.violation:
            print("Violation details:")
            print(f"- Message: {response.violation.user_message}")
            print(f"- Level: {response.violation.violation_level}")
            print(f"- Metadata: {response.violation.metadata}")

            assert "detector_id" in response.violation.metadata
            assert response.violation.metadata["detector_id"] == expected_detector
            assert "detection_type" in response.violation.metadata
            assert "score" in response.violation.metadata

        print("-------------------------------------")


if __name__ == "__main__":
    asyncio.run(test_fms_detectors_live())
