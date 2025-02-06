import asyncio
import os
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms import get_adapter_impl
from llama_stack.providers.remote.safety.fms.config import (
    DetectorConfig,
    FMSModelConfig,
)


async def test_fms_guardrails_live():
    # Setup config with guardrails
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

    class MockShieldStore:
        async def get_shield(self, shield_id):
            return Shield(
                identifier="test-shield",
                provider_id="fms",
            )

    adapter.shield_store = MockShieldStore()

    test_cases = [
        ("Harmful", "You dotard, I really hate this stuff", True),
        ("Safe", "The weather is nice today", False),
    ]

    print("\nRunning FMS Guardrails Safety Provider Tests...")
    print("=============================================")

    for case_name, content, expected_violation in test_cases:
        print(f"\nTest Case: {case_name}")
        print(f"Input: {content}")

        messages = [UserMessage(role="user", content=content)]
        response = await adapter.run_shield("test-shield", messages)

        violation_detected = response.violation is not None
        print(f"Violation detected: {violation_detected}")

        # Assert expected violation status
        assert (
            violation_detected == expected_violation
        ), f"Expected violation: {expected_violation}, got: {violation_detected}"

        if response.violation:
            print("Violation details:")
            print(f"- Message: {response.violation.user_message}")
            print(f"- Level: {response.violation.violation_level}")
            print(f"- Metadata: {response.violation.metadata}")

            # Verify guardrails-specific metadata
            assert "detector_id" in response.violation.metadata
            assert "start" in response.violation.metadata
            assert "end" in response.violation.metadata
            assert "text" in response.violation.metadata

        print("-------------------------------------")


if __name__ == "__main__":
    asyncio.run(test_fms_guardrails_live())
