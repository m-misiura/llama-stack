from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from llama_stack.apis.inference import (
    CompletionMessage,
    Message,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.remote.safety.fms.config import (
    BaseDetectorConfig,
    EndpointType,
)

# Configure logging
logger = logging.getLogger(__name__)


# Custom exceptions
class DetectorError(Exception):
    """Base exception for detector errors"""

    pass


class DetectorConfigError(DetectorError):
    """Configuration related errors"""

    pass


class DetectorRequestError(DetectorError):
    """HTTP request related errors"""

    pass


class DetectorValidationError(DetectorError):
    """Validation related errors"""

    pass


# Type aliases
MessageDict = Dict[str, Any]
DetectorResponse = Dict[str, Any]
Headers = Dict[str, str]
RequestPayload = Dict[str, Any]


class MessageTypes(Enum):
    """Message type constants"""

    USER = auto()
    SYSTEM = auto()
    TOOL = auto()
    COMPLETION = auto()

    @classmethod
    def to_str(cls, value: MessageTypes) -> str:
        """Convert enum to string representation"""
        return value.name.lower()


@dataclass(frozen=True)
class DetectionResult:
    """Structured detection result"""

    detection: str
    detection_type: str
    score: float
    detector_id: str
    text: str = ""
    start: int = 0
    end: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "detection": self.detection,
            "detection_type": self.detection_type,
            "score": self.score,
            "detector_id": self.detector_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class BaseDetector(Safety, ShieldsProtocolPrivate, ABC):
    """Base class for all safety detectors"""

    # Class constants
    DEFAULT_TIMEOUT: ClassVar[float] = 30.0
    MAX_RETRIES: ClassVar[int] = 3
    BACKOFF_FACTOR: ClassVar[float] = 1.5
    VALID_SCHEMES: ClassVar[set] = {"http", "https"}

    def __init__(self, config: BaseDetectorConfig) -> None:
        """Initialize detector with configuration"""
        self.config = config
        self.registered_shields: List[Shield] = []
        self.score_threshold: float = config.confidence_threshold
        self._http_client: Optional[httpx.AsyncClient] = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate detector configuration"""
        if not self.config:
            raise DetectorConfigError("Configuration is required")
        if not isinstance(self.config, BaseDetectorConfig):
            raise DetectorConfigError(f"Invalid config type: {type(self.config)}")

    async def initialize(self) -> None:
        """Initialize detector resources"""
        logger.info(f"Initializing {self.__class__.__name__}")
        self._http_client = httpx.AsyncClient(
            timeout=self.DEFAULT_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def shutdown(self) -> None:
        """Clean up detector resources"""
        logger.info(f"Shutting down {self.__class__.__name__}")
        if self._http_client:
            await self._http_client.aclose()

    async def register_shield(self, shield: Shield) -> None:
        """Register a shield with the detector"""
        if not shield or not shield.identifier:
            raise DetectorValidationError("Invalid shield configuration")
        logger.info(f"Registering shield {shield.identifier}")
        self.registered_shields.append(shield)

    def _should_process_message(self, message: Message) -> bool:
        """Check if this detector should process the given message type"""
        message_type_map = {
            MessageTypes.to_str(MessageTypes.USER): lambda m: isinstance(
                m, UserMessage
            ),
            MessageTypes.to_str(MessageTypes.SYSTEM): lambda m: isinstance(
                m, SystemMessage
            ),
            MessageTypes.to_str(MessageTypes.TOOL): lambda m: isinstance(
                m, ToolResponseMessage
            ),
            MessageTypes.to_str(MessageTypes.COMPLETION): lambda m: isinstance(
                m, CompletionMessage
            ),
        }

        return any(
            message_type_map.get(message_type, lambda _: False)(message)
            for message_type in self.config.message_types
        )

    def _filter_messages(self, messages: List[Message]) -> List[Message]:
        """Filter messages based on configured message types"""
        return [msg for msg in messages if self._should_process_message(msg)]

    def _validate_url(self, url: str) -> None:
        """Validate URL format"""
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise DetectorConfigError(f"Invalid URL format: {url}")
        if parsed.scheme not in self.VALID_SCHEMES:
            raise DetectorConfigError(f"Invalid URL scheme: {parsed.scheme}")

    def _construct_url(self) -> str:
        """Construct API URL based on configuration"""
        if self.config.use_orchestrator_api:
            if not self.config.orchestrator_base_url:
                raise DetectorConfigError(
                    "orchestrator_base_url is required when use_orchestrator_api is True"
                )
            base_url = self.config.orchestrator_base_url
            endpoint_info = (
                EndpointType.ORCHESTRATOR_CHAT.value
                if self.config.is_chat
                else EndpointType.ORCHESTRATOR_CONTENT.value
            )
        else:
            if not self.config.base_url:
                raise DetectorConfigError(
                    "base_url is required when use_orchestrator_api is False"
                )
            base_url = self.config.base_url
            endpoint_info = (
                EndpointType.DIRECT_CHAT.value
                if self.config.is_chat
                else EndpointType.DIRECT_CONTENT.value
            )

        url = f"{base_url.rstrip('/')}{endpoint_info['path']}"
        self._validate_url(url)
        logger.debug(
            f"Constructed URL: {url} for {'chat' if self.config.is_chat else 'content'} endpoint"
        )
        return url

    def _prepare_headers(self) -> Headers:
        """Prepare request headers based on configuration"""
        headers: Headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        if not self.config.use_orchestrator_api and self.config.detector_id:
            headers["detector-id"] = self.config.detector_id

        return headers

    def _prepare_request_payload(
        self, messages: List[Message], params: Optional[Dict[str, Any]] = None
    ) -> RequestPayload:
        """Prepare request payload based on endpoint type and orchestrator mode"""
        detector_params = {}
        if self.config.detector_params:
            detector_params = {
                k: v
                for k, v in vars(self.config.detector_params).items()
                if v is not None
            }

        if self.config.use_orchestrator_api:
            payload: RequestPayload = {
                "detectors": {self.config.detector_id: detector_params}
            }
            if self.config.is_chat:
                payload["messages"] = [msg.dict() for msg in messages]
            else:
                payload["content"] = messages[0].content
        else:
            if self.config.is_chat:
                payload = {
                    "messages": [msg.dict() for msg in messages],
                    "detector_params": (
                        detector_params if detector_params else params or {}
                    ),
                }
            else:
                payload = {
                    "contents": [msg.content for msg in messages],
                    "detector_params": (
                        detector_params if detector_params else params or {}
                    ),
                }

        return payload

    async def _make_request(
        self,
        request: RequestPayload,
        headers: Optional[Headers] = None,
        timeout: Optional[float] = None,
    ) -> DetectorResponse:
        """Make HTTP request with error handling and retries"""
        if not self._http_client:
            raise DetectorError("HTTP client not initialized")

        url = self._construct_url()
        default_headers = self._prepare_headers()
        headers = {**default_headers, **(headers or {})}

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self._http_client.post(
                    url,
                    json=request,
                    headers=headers,
                    timeout=timeout or self.DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error occurred (attempt {attempt + 1}/{self.MAX_RETRIES}): {e.response.text}"
                )
                if attempt == self.MAX_RETRIES - 1:
                    raise DetectorRequestError(
                        f"API Error after {self.MAX_RETRIES} attempts: {e.response.text}"
                    ) from e

            except httpx.RequestError as e:
                logger.error(
                    f"Request error occurred (attempt {attempt + 1}/{self.MAX_RETRIES}): {str(e)}"
                )
                if attempt == self.MAX_RETRIES - 1:
                    raise DetectorRequestError(
                        f"Request Error after {self.MAX_RETRIES} attempts: {str(e)}"
                    ) from e

            # Exponential backoff
            await asyncio.sleep(self.BACKOFF_FACTOR**attempt)

    def _process_detection(
        self, detection: Dict[str, Any]
    ) -> Optional[DetectionResult]:
        """Process detection result and validate against threshold"""
        if not detection.get("score"):
            logger.warning("Detection missing score field")
            return None

        score = detection.get("score", 0)
        if score > self.score_threshold:
            return DetectionResult(
                detection="Yes",
                detection_type=detection["detection_type"],
                score=score,
                detector_id=detection.get("detector_id", self.config.detector_id),
                text=detection.get("text", ""),
                start=detection.get("start", 0),
                end=detection.get("end", 0),
                metadata=detection.get("metadata"),
            )
        return None

    def create_violation_response(
        self,
        detection: DetectionResult,
        detector_id: str,
        level: ViolationLevel = ViolationLevel.ERROR,
    ) -> RunShieldResponse:
        """Create standardized violation response"""
        return RunShieldResponse(
            violation=SafetyViolation(
                user_message=f"Content flagged by {detector_id} as {detection.detection_type} with confidence {detection.score:.2f}",
                violation_level=level,
                metadata=detection.to_dict(),
            )
        )

    def _validate_shield(self, shield: Shield) -> None:
        """Validate shield configuration"""
        if not shield:
            raise DetectorValidationError("Shield not found")
        if not shield.identifier:
            raise DetectorValidationError("Shield missing identifier")

    @abstractmethod
    async def _run_shield_impl(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Implementation specific shield running logic"""
        pass

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Run safety checks using configured shield"""
        if not messages:
            logger.debug("No messages provided")
            return RunShieldResponse(violation=None)

        filtered_messages = self._filter_messages(messages)
        if not filtered_messages:
            logger.debug(
                f"No messages of configured types {self.config.message_types} to process"
            )
            return RunShieldResponse(violation=None)

        return await self._run_shield_impl(shield_id, filtered_messages, params)


class DetectorProvider:
    """Provider that manages multiple detectors and allows running them all at once"""

    def __init__(self, detectors: Dict[str, BaseDetector]) -> None:
        """Initialize provider with detectors"""
        if not detectors:
            raise DetectorConfigError("At least one detector must be provided")
        self.detectors = detectors

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
    ) -> Dict[str, RunShieldResponse]:
        """Run all detectors and return results from each"""
        if not shield_id:
            raise DetectorValidationError("Shield ID is required")
        if not messages:
            return {}

        results = {}
        for name, detector in self.detectors.items():
            try:
                response = await detector.run_shield(shield_id, messages)
                results[name] = response
            except Exception as e:
                logger.error(f"Error running detector {name}: {str(e)}")
                results[name] = RunShieldResponse(
                    violation=SafetyViolation(
                        user_message=f"Detector {name} failed: {str(e)}",
                        violation_level=ViolationLevel.ERROR,
                    )
                )
        return results

    async def shutdown(self) -> None:
        """Shutdown all detectors"""
        for detector in self.detectors.values():
            try:
                await detector.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down detector: {str(e)}")
