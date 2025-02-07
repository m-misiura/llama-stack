import asyncio
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms import get_adapter_impl
from llama_stack.providers.remote.safety.fms.config import (
    BaseContentDetectionConfig,
    ContentDetectionConfig,
)


async def test_fms_live():
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

    # Initialize adapter
    adapter = await get_adapter_impl(config, None)

    # Mock shield store with all required fields
    class MockShieldStore:
        async def get_shield(self, shield_id):
            return Shield(
                identifier="test-shield",
                provider_id="fms",
            )

    adapter.shield_store = MockShieldStore()

    # Test cases
    test_cases = [
        ("Harmful", "You dotard, I really hate this stuff"),
        ("Safe", "The weather is nice today"),
    ]

    print("\nRunning FMS Safety Provider Tests...")
    print("=====================================")

    for case_name, content in test_cases:
        print(f"\nTest Case: {case_name}")
        print(f"Input: {content}")

        messages = [UserMessage(role="user", content=content)]
        response = await adapter.run_shield("test-shield", messages)

        print(f"Violation detected: {response.violation is not None}")
        if response.violation:
            print("Violation details:")
            print(f"- Message: {response.violation.user_message}")
            print(f"- Level: {response.violation.violation_level}")
            print(f"- Metadata: {response.violation.metadata}")
        print("-------------------------------------")


if __name__ == "__main__":
    asyncio.run(test_fms_live())
