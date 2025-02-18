from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union
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

    detector_id: str
    is_chat: bool = False
    base_url: Optional[str] = None
    orchestrator_base_url: Optional[str] = None
    confidence_threshold: float = 0.5
    use_orchestrator_api: bool = False
    detector_params: Optional[DetectorParams] = None

    def validate(self):
        """Validate configuration after all settings are propagated"""
        if not self.use_orchestrator_api and not self.base_url:
            raise ValueError("base_url is required when use_orchestrator_api is False")
        if self.use_orchestrator_api and not self.orchestrator_base_url:
            raise ValueError(
                "orchestrator_base_url is required when use_orchestrator_api is True"
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


@json_schema_type
@dataclass
class ChatDetectorConfig(BaseDetectorConfig):
    """Configuration for chat detectors"""

    def __post_init__(self):
        self.is_chat = True


@json_schema_type
@dataclass
class FMSSafetyProviderConfig:
    """Configuration for the FMS Safety Provider"""

    detectors: Dict[str, Union[ChatDetectorConfig, ContentDetectorConfig]]
    orchestrator_base_url: Optional[str] = None
    use_orchestrator_api: bool = False  # New field to control orchestrator mode

    def __post_init__(self):
        """Validate and propagate orchestrator configuration"""

        # Check for mixed API usage
        mixed_api_detectors = {
            detector_id: detector.use_orchestrator_api
            for detector_id, detector in self.detectors.items()
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

        if self.use_orchestrator_api:
            if not self.orchestrator_base_url:
                raise ValueError(
                    "orchestrator_base_url is required when use_orchestrator_api is True"
                )
            # Check for invalid base_url configurations
            invalid_detectors = [
                detector_id
                for detector_id, detector in self.detectors.items()
                if detector.base_url is not None
            ]

            if invalid_detectors:
                raise ValueError(
                    f"When using orchestrator API, base_url should not be specified for detectors: {invalid_detectors}. "
                    "All requests will be routed through the orchestrator_base_url."
                )

            # Propagate orchestrator settings to all detectors
            for detector in self.detectors.values():
                object.__setattr__(detector, "use_orchestrator_api", True)
                object.__setattr__(
                    detector, "orchestrator_base_url", self.orchestrator_base_url
                )
