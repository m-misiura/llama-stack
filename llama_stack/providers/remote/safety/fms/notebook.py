# ruff: noqa
# %% Imports and setup
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from llama_stack.apis.inference import (
    UserMessage,
    SystemMessage,
    ToolResponseMessage,
    CompletionMessage,
    Message,
)
from llama_stack.apis.safety import RunShieldResponse
from llama_stack.providers.remote.safety.fms import get_adapter_impl
from llama_stack.providers.remote.safety.fms.config import (
    ContentDetectorConfig,
    DetectorParams,
    FMSSafetyProviderConfig,
    ChatDetectorConfig,
)
from llama_stack.apis.inference import StopReason

# Configure logging with better formatting
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# %% Test infrastructure
@dataclass(frozen=True)
class Shield:
    """Test shield configuration"""

    identifier: str
    provider_id: str = field(default="fms-safety")  # Match real provider ID
    provider_resource_id: str = field(init=False)
    type: str = field(default="shield")
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "provider_resource_id", self.identifier)


class MockShieldStore:
    """Mock shield storage for testing"""

    def __init__(self):
        self._shields = {}
        self._detector_configs = {}

    def register_detector_config(self, detector_id: str, config: Any) -> None:
        """Register detector configuration"""
        self._detector_configs[detector_id] = config

    async def get_shield(self, shield_id: str) -> Shield:
        """Get or create shield by identifier"""
        if shield_id not in self._shields:
            config = self._detector_configs.get(shield_id)
            self._shields[shield_id] = Shield(
                identifier=shield_id, params={"regex": ["email"]} if config else {}
            )
        return self._shields[shield_id]


# %% Helper functions
async def setup_detector(config: FMSSafetyProviderConfig):
    """Setup detector with shield store"""
    detectors = await get_adapter_impl(config)
    shield_store = MockShieldStore()

    # Register configs first
    for detector_id, detector_config in config.detectors.items():
        shield_store.register_detector_config(detector_id, detector_config)

    # Then update detectors
    for detector in detectors.detectors.values():
        detector.shield_store = shield_store

    return detectors


async def run_test(
    detectors, messages: List[Message], shield_id: str = "regex"
):  # Changed default
    """Run test and print results"""
    response = await detectors.run_shield(shield_id, messages)
    print(f"Shield response: {response}")
    return response


# %% Example 1: Basic Content Detection
logger.info("Running basic content detection test")
basic_config = FMSSafetyProviderConfig(
    detectors={
        "regex": ContentDetectorConfig(
            detector_id="regex",
            base_url="http://regex-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
            detector_params=DetectorParams(regex=["email"]),
        )
    }
)

messages = [
    UserMessage(content="Your email is test@ibm.com"),
    SystemMessage(content="System message with test@ibm.com"),
]

basic_detectors = await setup_detector(basic_config)
await run_test(basic_detectors, messages)

# %% Example 2: Tool Message Detection
logger.info("Running tool message detection test")
tool_config = FMSSafetyProviderConfig(
    detectors={
        "regex": ContentDetectorConfig(
            detector_id="regex",
            base_url="http://regex-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
            detector_params=DetectorParams(regex=["email"]),
            message_types={"tool"},
        )
    }
)

tool_messages = [
    ToolResponseMessage(
        call_id="test-call-123",
        tool_name="code_interpreter",
        content="The script executed successfully. Contact: test@ibm.com",
    ),
]

tool_detectors = await setup_detector(tool_config)
await run_test(tool_detectors, tool_messages)

# %% Example 3: Completion Message Detection
logger.info("Running completion message detection test")
completion_config = FMSSafetyProviderConfig(
    detectors={
        "regex": ContentDetectorConfig(
            detector_id="regex",
            base_url="http://regex-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
            detector_params=DetectorParams(regex=["email"]),
            message_types={"completion"},
        )
    }
)

completion_messages = [
    CompletionMessage(
        content="Here's my email: test@ibm.com",
        stop_reason=StopReason.end_of_turn,
    ),
]

completion_detectors = await setup_detector(completion_config)
await run_test(completion_detectors, completion_messages)

# %% Example 4: Multi-Detector Configuration
logger.info("Running multi-detector test")
multi_detector_config = FMSSafetyProviderConfig(
    detectors={
        "regex": ContentDetectorConfig(
            detector_id="regex",
            base_url="http://regex-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
            detector_params=DetectorParams(regex=["email"]),
            message_types={"system"},
        ),
        "hap": ContentDetectorConfig(
            detector_id="hap",
            base_url="http://hap-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
        ),
        "chat": ChatDetectorConfig(
            detector_id="granite",
            base_url="http://granite-detector-route-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
            detector_params=DetectorParams(
                temperature=0.0,
                risk_name="Star Wars",
                risk_definition="The message contains references to Star Wars, especially Luke Skywalker.",
            ),
            message_types={"system", "completion"},
        ),
    }
)

mixed_messages = [
    # UserMessage(content="Your email is test@ibm.com"),
    SystemMessage(
        content="Luke Skywalker is a Jedi and Darth Vader is his father. Luke Skywalker loves lightsabers."
    ),
]

multi_detectors = await setup_detector(multi_detector_config)
await run_test(multi_detectors, mixed_messages)

# %% Example 5: Orchestrator Configuration
logger.info("Running orchestrator configuration test")
orchestrator_config = FMSSafetyProviderConfig(
    orchestrator_base_url="https://guardrails-nlp-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
    use_orchestrator_api=True,
    detectors={
        "regex": ContentDetectorConfig(
            detector_id="regex",
            detector_params=DetectorParams(regex=["email"]),
            message_types={"user", "system"},
        ),
        "chat": ChatDetectorConfig(
            detector_id="granite",
            detector_params=DetectorParams(
                temperature=0.0,
                risk_name="code_execution",
                risk_definition="The message requests or attempts to execute potentially harmful or unauthorized code that could compromise system security or perform malicious actions.",
            ),
            message_types={"user"},
        ),
    },
)

orchestrator_messages = [
    UserMessage(content="Your email is test@ibm.com. Please execute rm -rf /"),
    SystemMessage(content="System message with test@ibm.com"),
]

orch_detectors = await setup_detector(orchestrator_config)
await run_test(orch_detectors, orchestrator_messages)


# %% Example 6: Multi-Detector Configuration with Different Thresholds
logger.info("Running multi-threshold detector test")
threshold_config = FMSSafetyProviderConfig(
    orchestrator_base_url="https://guardrails-nlp-test.apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
    use_orchestrator_api=True,
    detectors={
        "regex": ContentDetectorConfig(
            detector_id="regex",
            detector_params=DetectorParams(regex=["email"]),
            message_types={"user", "system"},
        ),
        "chat": ChatDetectorConfig(
            detector_id="granite",
            detector_params=DetectorParams(
                temperature=0.0,
                risk_name="code_execution",
                risk_definition="The message requests or attempts to execute potentially harmful or unauthorized code that could compromise system security or perform malicious actions.",
            ),
            message_types={"user"},
            confidence_threshold=0.3,  # Lower threshold for code execution risks
        ),
    },
)

# Test messages that should trigger different thresholds
threshold_messages = [
    # UserMessage(
    #     content="Your email is definitely test@ibm.com"
    # ),  # Should trigger regex
    # SystemMessage(
    #     content="Consider running sudo rm -rf /"
    # ),  # Should trigger code execution
    UserMessage(content="plz remotely execute this code"),  # Borderline case
]

threshold_detectors = await setup_detector(threshold_config)
await run_test(threshold_detectors, threshold_messages)


# %% Error handling examples
async def test_error_cases():
    """Test various error cases"""
    try:
        # Invalid detector configuration
        invalid_config = FMSSafetyProviderConfig(
            detectors={
                "regex": ContentDetectorConfig(
                    detector_id="regex",
                    message_types={"invalid_type"},
                )
            }
        )
        await setup_detector(invalid_config)
    except Exception as e:
        logger.error(f"Expected error with invalid message type: {e}")

    try:
        # Missing required URL
        missing_url_config = FMSSafetyProviderConfig(
            detectors={
                "regex": ContentDetectorConfig(
                    detector_id="regex",
                )
            }
        )
        await setup_detector(missing_url_config)
    except Exception as e:
        logger.error(f"Expected error with missing URL: {e}")


await test_error_cases()
