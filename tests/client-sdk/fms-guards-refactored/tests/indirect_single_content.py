import asyncio
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms_refactor import get_adapter_impl
from llama_stack.providers.remote.safety.fms_refactor.config import (
    ContentDetectorConfig,
)


async def test_regex_detector_orchestrator():
    """Test regex detector with email pattern via orchestrator"""
    config = ContentDetectorConfig(
        base_url="https://guardrails-nlp-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
        detector_id="regex",
        confidence_threshold=0.5,
        use_orchestrator_api=True,  # Enable orchestrator API
        guardrails_detectors={
            "regex": {  # Regex detector config
                "regex": ["email"],
            },
        },
    )

    adapter = None
    try:
        adapter = await get_adapter_impl(config)

        class MockShieldStore:
            async def get_shield(self, shield_id: str) -> Shield:
                return Shield(
                    identifier="test-shield-regex",
                    provider_id="content",
                    provider_resource_id="regex",
                    params={},
                )

        adapter.shield_store = MockShieldStore()

        test_content = "Your email is test@ibm.com"
        messages = [UserMessage(role="user", content=test_content)]

        print("\nRunning Regex Detector Orchestrator Test")
        print("=====================================")
        print(f"Input: {test_content}")

        response = await adapter.run_shield("test-shield-regex", messages)
        violation_detected = response.violation is not None
        print(f"Violation detected: {violation_detected}")

        assert violation_detected, "Expected email detection violation"

        if response.violation:
            print("Violation details:")
            print(f"- Message: {response.violation.user_message}")
            print(f"- Level: {response.violation.violation_level}")
            print(f"- Metadata: {response.violation.metadata}")

    finally:
        if adapter:
            await adapter.shutdown()


if __name__ == "__main__":
    asyncio.run(test_regex_detector_orchestrator())
