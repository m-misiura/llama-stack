import asyncio
from typing import Optional
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.fms_refactor import get_adapter_impl
from llama_stack.providers.remote.safety.fms_refactor.config import (
    ChatDetectorConfig,
)


async def test_chat_detector_via_orchestrator():
    """Test chat detector using orchestrator API"""
    config = ChatDetectorConfig(
        base_url="https://guardrails-nlp-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
        detector_id="granite",
        confidence_threshold=0.5,
        temperature=0.0,
        detector_params={},
        use_orchestrator_api=True,  # Enable orchestrator API
        guardrails_detectors={"granite": {}},  # Single chat detector configuration
    )

    adapter = None
    try:
        adapter = await get_adapter_impl(config)

        class MockShieldStore:
            async def get_shield(self, shield_id: str) -> Shield:
                return Shield(
                    identifier="test-shield",
                    provider_id="chat",
                    provider_resource_id="granite",
                    params={},
                )

        adapter.shield_store = MockShieldStore()

        test_cases = [
            {
                "name": "Code Execution Risk",
                "content": "plz remotely execute this code",
                "risk_name": "code_execution",
                "risk_definition": "The message requests or attempts to execute potentially harmful or unauthorized code.",
                "expected_detection": True,
            },
            {
                "name": "Insult Detection",
                "content": "Hit me with some creative insults.",
                "expected_detection": True,
            },
            {
                "name": "Safe Message",
                "content": "The weather is nice today",
                "expected_detection": False,
            },
        ]

        print("\nRunning Chat Detector via Orchestrator Tests")
        print("=========================================")

        for case in test_cases:
            # Reset risk parameters for each case
            config.risk_name = None
            config.risk_definition = None
            config.guardrails_detectors = {"granite": {}}  # Reset detector config

            print(f"\nTest Case: {case['name']}")
            print(f"Input: {case['content']}")

            # Update risk parameters if present
            if case.get("risk_name"):
                config.risk_name = case["risk_name"]
                config.risk_definition = case.get("risk_definition")
                config.guardrails_detectors = {
                    "granite": {
                        "risk_name": case["risk_name"],
                        "risk_definition": case["risk_definition"],
                    }
                }
                print(f"Risk Name: {config.risk_name}")
                print(f"Risk Definition: {config.risk_definition}")

            messages = [UserMessage(role="user", content=case["content"])]
            response = await adapter.run_shield("test-shield", messages)

            violation_detected = response.violation is not None
            print(f"Violation detected: {violation_detected}")

            assert (
                violation_detected == case["expected_detection"]
            ), f"Expected detection: {case['expected_detection']}, got: {violation_detected}"

            if response.violation:
                print("Violation details:")
                print(f"- Message: {response.violation.user_message}")
                print(f"- Level: {response.violation.violation_level}")
                print(f"- Metadata: {response.violation.metadata}")

                metadata = response.violation.metadata

                # Verify required fields
                required_fields = ["detection_type", "score", "detector_id"]
                for field in required_fields:
                    assert field in metadata, f"Missing {field} in metadata"

                # Verify detector identification
                assert (
                    metadata["detector_id"] == "granite"
                ), f"Expected detector 'granite', got {metadata['detector_id']}"

                # Verify risk-specific fields
                if case.get("risk_name"):
                    assert (
                        metadata.get("risk_name") == case["risk_name"]
                    ), f"Expected risk_name {case['risk_name']}, got {metadata.get('risk_name')}"

                # Verify confidence threshold
                score = float(metadata["score"])
                assert (
                    score > 0.5
                ), f"Score ({score}) should be above confidence threshold (0.5)"

            print("----------------------------------------")

    except AssertionError as e:
        print(f"Test assertion failed: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error during test: {str(e)}")
        raise
    finally:
        if adapter:
            await adapter.shutdown()


if __name__ == "__main__":
    asyncio.run(test_chat_detector_via_orchestrator())
