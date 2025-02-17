from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Set
from llama_models.schema_utils import json_schema_type


class EndpointType(Enum):
    """API endpoint types and their paths"""

    DIRECT_CONTENT = {
        "path": "/api/v1/text/contents",
        "version": "v1",
        "type": "content",
    }
    DIRECT_CHAT = {"path": "/api/v1/text/chat", "version": "v1", "type": "chat"}
    ORCHESTRATOR_CONTENT = {
        "path": "/api/v2/text/detection/content",
        "version": "v2",
        "type": "content",
    }
    ORCHESTRATOR_CHAT = {
        "path": "/api/v2/text/detection/chat",
        "version": "v2",
        "type": "chat",
    }

    @property
    def path(self) -> str:
        """Get endpoint path"""
        return self.value["path"]

    @property
    def version(self) -> str:
        """Get API version"""
        return self.value["version"]

    @property
    def type(self) -> str:
        """Get endpoint type"""
        return self.value["type"]

    @classmethod
    def get_endpoint(cls, is_orchestrator: bool, is_chat: bool) -> "EndpointType":
        """Get the appropriate endpoint type based on configuration"""
        if is_orchestrator:
            return cls.ORCHESTRATOR_CHAT if is_chat else cls.ORCHESTRATOR_CONTENT
        return cls.DIRECT_CHAT if is_chat else cls.DIRECT_CONTENT


@json_schema_type
@dataclass
class DetectorParams:
    """Common detector parameters"""

    regex: Optional[List[str]] = None
    temperature: Optional[float] = None
    risk_name: Optional[str] = None
    risk_definition: Optional[str] = None


@json_schema_type
@dataclass
class BaseDetectorConfig:
    """Base configuration for all detectors"""

    VALID_MESSAGE_TYPES = {"user", "system", "tool", "completion"}

    detector_id: str
    is_chat: bool = False
    base_url: Optional[str] = None
    orchestrator_base_url: Optional[str] = None
    confidence_threshold: float = 0.5
    use_orchestrator_api: bool = False
    detector_params: Optional[DetectorParams] = None
    message_types: Set[str] = field(
        default_factory=lambda: {"user", "system", "tool", "completion"}
    )

    def validate(self):
        """Validate configuration after all settings are propagated"""
        # Validate message types
        invalid_types = self.message_types - self.VALID_MESSAGE_TYPES
        if invalid_types:
            raise ValueError(
                f"Invalid message types: {invalid_types}. "
                f"Valid types are: {self.VALID_MESSAGE_TYPES}"
            )

        # Validate URL configuration
        if not self.use_orchestrator_api and not self.base_url:
            raise ValueError("base_url is required when use_orchestrator_api is False")
        if self.use_orchestrator_api and not self.orchestrator_base_url:
            raise ValueError(
                "orchestrator_base_url is required when use_orchestrator_api is True"
            )

    def __post_init__(self):
        """Set chat mode and validate message types immediately"""
        # Only validate message types in post_init as they don't depend on orchestrator settings
        invalid_types = self.message_types - self.VALID_MESSAGE_TYPES
        if invalid_types:
            raise ValueError(
                f"Invalid message types: {invalid_types}. "
                f"Valid types are: {self.VALID_MESSAGE_TYPES}"
            )

    @property
    def endpoint_type(self) -> EndpointType:
        """Get endpoint type based on configuration"""
        return EndpointType.get_endpoint(self.use_orchestrator_api, self.is_chat)


@json_schema_type
@dataclass
class ContentDetectorConfig(BaseDetectorConfig):
    """Configuration for content detectors"""

    allow_list: Optional[List[str]] = None
    block_list: Optional[List[str]] = None

    def __post_init__(self):
        self.is_chat = False
        super().__post_init__()  # Call parent's validation


@json_schema_type
@dataclass
class ChatDetectorConfig(BaseDetectorConfig):
    """Configuration for chat detectors"""

    def __post_init__(self):
        self.is_chat = True
        super().__post_init__()  # Call parent's validation


@json_schema_type
@dataclass
class FMSSafetyProviderConfig:
    """Configuration for the FMS Safety Provider organized by message types"""

    detectors: Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]
    orchestrator_base_url: Optional[str] = None
    use_orchestrator_api: bool = False

    @property
    def all_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get all detectors"""
        return self.detectors

    @property
    def user_message_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors configured for user messages"""
        return {
            detector_id: detector
            for detector_id, detector in self.detectors.items()
            if "user" in detector.message_types
        }

    @property
    def system_message_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors configured for system messages"""
        return {
            detector_id: detector
            for detector_id, detector in self.detectors.items()
            if "system" in detector.message_types
        }

    @property
    def tool_response_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors configured for tool responses"""
        return {
            detector_id: detector
            for detector_id, detector in self.detectors.items()
            if "tool" in detector.message_types
        }

    @property
    def completion_message_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors configured for completion messages"""
        return {
            detector_id: detector
            for detector_id, detector in self.detectors.items()
            if "completion" in detector.message_types
        }

    def validate_mixed_api_usage(self):
        """Check for mixed API usage across all detector types"""
        mixed_api_detectors = {
            detector_id: detector.use_orchestrator_api
            for detector_id, detector in self.all_detectors.items()
        }

        orchestrator_detectors = [
            d_id for d_id, uses_orch in mixed_api_detectors.items() if uses_orch
        ]
        direct_detectors = [
            d_id for d_id, uses_orch in mixed_api_detectors.items() if not uses_orch
        ]

        if orchestrator_detectors and direct_detectors:
            raise ValueError(
                "Mixed API usage detected. All detectors must use either direct or orchestrator API:\n"
                f"- Orchestrator API detectors: {orchestrator_detectors}\n"
                f"- Direct API detectors: {direct_detectors}\n"
                "Please configure all detectors consistently."
            )

    def validate_orchestrator_config(self):
        """Validate orchestrator configuration"""
        if self.use_orchestrator_api:
            if not self.orchestrator_base_url:
                raise ValueError(
                    "orchestrator_base_url is required when use_orchestrator_api is True"
                )

            # Check for invalid base_url configurations
            invalid_detectors = [
                detector_id
                for detector_id, detector in self.all_detectors.items()
                if detector.base_url is not None
            ]

            if invalid_detectors:
                raise ValueError(
                    f"When using orchestrator API, base_url should not be specified for detectors: {invalid_detectors}. "
                    "All requests will be routed through the orchestrator_base_url."
                )

    def propagate_orchestrator_settings(self):
        """Propagate orchestrator settings to all detectors"""
        if self.use_orchestrator_api:
            for detector in self.all_detectors.values():
                object.__setattr__(detector, "use_orchestrator_api", True)
                object.__setattr__(
                    detector, "orchestrator_base_url", self.orchestrator_base_url
                )

    def __post_init__(self):
        """Validate and propagate orchestrator configuration"""
        self.validate_mixed_api_usage()
        self.validate_orchestrator_config()
        self.propagate_orchestrator_settings()

        # Validate all detector configs after propagation
        for detector in self.all_detectors.values():
            detector.validate()
