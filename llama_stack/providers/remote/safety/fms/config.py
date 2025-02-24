from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
from urllib.parse import urlparse

from llama_stack.schema_utils import json_schema_type


def resolve_detector_config(
    data: Dict[str, Any]
) -> Union["ContentDetectorConfig", "ChatDetectorConfig"]:
    """Resolve detector configuration from dictionary."""
    if isinstance(data, (ContentDetectorConfig, ChatDetectorConfig)):
        return data

    # Convert detector_params if present
    if "detector_params" in data and isinstance(data["detector_params"], dict):
        params = data["detector_params"]
        if "detectors" in params:
            # Handle orchestrator mode
            data["detector_params"] = DetectorParams(detectors=params["detectors"])
        else:
            # Handle direct mode
            data["detector_params"] = DetectorParams(**params)

    # Determine detector type
    if data.get("is_chat", False):
        return ChatDetectorConfig(**data)
    return ContentDetectorConfig(**data)


class MessageType(Enum):
    """Valid message types for detectors"""

    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"
    COMPLETION = "completion"

    @classmethod
    def as_set(cls) -> Set[str]:
        """Get all valid message types as a set"""
        return {member.value for member in cls}


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
    detectors: Optional[Dict[str, Dict[str, Any]]] = None

    def validate(self) -> None:
        """Validate detector parameters"""
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")


@json_schema_type
@dataclass
class BaseDetectorConfig:
    """Base configuration for all detectors"""

    detector_id: str
    is_chat: bool = False
    base_url: Optional[str] = None
    orchestrator_base_url: Optional[str] = None
    confidence_threshold: float = 0.5
    use_orchestrator_api: bool = False
    detector_params: Optional[DetectorParams] = None
    message_types: Set[str] = field(default_factory=lambda: MessageType.as_set())
    auth_token: Optional[str] = None

    def _validate_urls(self) -> None:
        """Validate URL configurations"""
        if not self.use_orchestrator_api and not self.base_url:
            raise ValueError("base_url is required when use_orchestrator_api is False")
        if self.use_orchestrator_api and not self.orchestrator_base_url:
            raise ValueError(
                "orchestrator_base_url is required when use_orchestrator_api is True"
            )

        for url in [self.base_url, self.orchestrator_base_url]:
            if url:
                parsed = urlparse(url)
                if not all([parsed.scheme, parsed.netloc]):
                    raise ValueError(f"Invalid URL format: {url}")
                if parsed.scheme not in {"http", "https"}:
                    raise ValueError(f"URL must use http or https scheme: {url}")

    def _validate_message_types(self) -> None:
        """Validate message type configuration"""
        if isinstance(self.message_types, (list, tuple)):
            self.message_types = set(self.message_types)

        invalid_types = self.message_types - MessageType.as_set()
        if invalid_types:
            raise ValueError(
                f"Invalid message types: {invalid_types}. "
                f"Valid types are: {MessageType.as_set()}"
            )

    def validate(self) -> None:
        """Validate configuration after all settings are propagated"""
        self._validate_message_types()
        self._validate_urls()
        if self.detector_params:
            self.detector_params.validate()

    def __post_init__(self) -> None:
        """Validate configuration immediately after initialization"""
        self._validate_message_types()

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
        super().__post_init__()


@json_schema_type
@dataclass
class ChatDetectorConfig(BaseDetectorConfig):
    """Configuration for chat detectors"""

    def __post_init__(self):
        self.is_chat = True
        super().__post_init__()


@json_schema_type
@dataclass
class FMSSafetyProviderConfig:
    """Configuration for the FMS Safety Provider organized by message types"""

    detectors: Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]
    orchestrator_base_url: Optional[str] = None
    use_orchestrator_api: bool = False

    def __post_init__(self):
        """Convert detector dictionaries to proper config objects and validate"""
        # Convert dictionary detectors to config objects
        if isinstance(self.detectors, dict):
            converted_detectors = {}
            for k, v in self.detectors.items():
                if isinstance(v, dict):
                    # Ensure detector_id is set
                    if "detector_id" not in v:
                        v["detector_id"] = k
                    converted_detectors[k] = resolve_detector_config(v)
                else:
                    converted_detectors[k] = v
            self.detectors = converted_detectors

        # Run validations
        self.validate_mixed_api_usage()
        self.validate_orchestrator_config()
        self.propagate_orchestrator_settings()

        # Validate all detector configs
        for detector in self.all_detectors.values():
            detector.validate()

    def get_detectors_by_type(
        self, message_type: Union[str, MessageType]
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors configured for a specific message type"""
        type_value = (
            message_type.value
            if isinstance(message_type, MessageType)
            else message_type
        )
        return {
            detector_id: detector
            for detector_id, detector in self.detectors.items()
            if type_value in detector.message_types
        }

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
        return self.get_detectors_by_type(MessageType.USER)

    @property
    def system_message_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors configured for system messages"""
        return self.get_detectors_by_type(MessageType.SYSTEM)

    @property
    def tool_response_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors configured for tool responses"""
        return self.get_detectors_by_type(MessageType.TOOL)

    @property
    def completion_message_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors configured for completion messages"""
        return self.get_detectors_by_type(MessageType.COMPLETION)

    def _update_detector_settings(self, detector: BaseDetectorConfig) -> None:
        """Update detector settings with orchestrator configuration"""
        detector.use_orchestrator_api = True
        detector.orchestrator_base_url = self.orchestrator_base_url

    def propagate_orchestrator_settings(self) -> None:
        """Propagate orchestrator settings to all detectors"""
        if self.use_orchestrator_api:
            for detector in self.all_detectors.values():
                self._update_detector_settings(detector)

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
