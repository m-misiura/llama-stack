from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Dict, List, Optional, Tuple
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
    ShieldStore,
)
from llama_stack.apis.shields import Shield, Shields, ListShieldsResponse
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.remote.safety.fms.config import (
    BaseDetectorConfig,
    EndpointType,
    BaseDetectorConfig,
    ChatDetectorConfig,
    ContentDetectorConfig,
    DetectorParams,
    FMSSafetyProviderConfig,
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
        self._shield_store: Optional[ShieldStore] = SimpleShieldStore()  # Add this line
        self._validate_config()

    @property
    def shield_store(self) -> ShieldStore:
        """Get shield store instance"""
        return self._shield_store

    @shield_store.setter
    def shield_store(self, value: ShieldStore) -> None:
        """Set shield store instance"""
        self._shield_store = value

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

        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"

        return headers

    def _prepare_request_payload(
        self, messages: List[Message], params: Optional[Dict[str, Any]] = None
    ) -> RequestPayload:
        """Prepare request payload based on endpoint type and orchestrator mode"""
        if self.config.use_orchestrator_api:
            payload: RequestPayload = {}

            # Handle detector configuration
            if self.config.detector_params:
                # Case 1: New style with explicit detectors configuration
                if (
                    hasattr(self.config.detector_params, "detectors")
                    and self.config.detector_params.detectors
                ):
                    # Pass through detectors configuration as-is
                    payload["detectors"] = self.config.detector_params.detectors
                else:
                    # Legacy style: Convert any detector params to detector config
                    detector_config = {}
                    detector_params = {
                        k: v
                        for k, v in vars(self.config.detector_params).items()
                        if v is not None and k != "detectors"
                    }

                    if detector_params:
                        detector_config[self.config.detector_id] = detector_params

                    payload["detectors"] = detector_config

            # Add content or messages based on mode
            if self.config.is_chat:
                payload["messages"] = [msg.dict() for msg in messages]
            else:
                payload["content"] = messages[0].content

            logger.debug(f"Prepared orchestrator payload: {payload}")
            return payload
        else:
            # Handle direct mode (unchanged)
            detector_params = {}
            if self.config.detector_params:
                detector_params = {
                    k: v
                    for k, v in vars(self.config.detector_params).items()
                    if v is not None
                }

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
    ) -> Tuple[Optional[DetectionResult], float]:
        """Process detection result and return both result and score"""
        score = detection.get("score", 0.0)

        if not "score" in detection:
            logger.warning("Detection missing score field")
            return None, 0.0

        if score > self.score_threshold:
            return (
                DetectionResult(
                    detection="Yes",
                    detection_type=detection["detection_type"],
                    score=score,
                    detector_id=detection.get("detector_id", self.config.detector_id),
                    text=detection.get("text", ""),
                    start=detection.get("start", 0),
                    end=detection.get("end", 0),
                    metadata=detection.get("metadata"),
                ),
                score,
            )
        return None, score

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

        # Validate message types first
        filtered_messages = self._filter_messages(messages)
        if not filtered_messages:
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=f"Message type not supported. Shield {shield_id} only handles: {list(self.config.message_types)}",
                )
            )

        return await self._run_shield_impl(shield_id, filtered_messages, params)


class SimpleShieldStore(ShieldStore):
    """Simplified shield store with caching"""

    def __init__(self):
        self._shields = {}
        self._detector_configs = {}
        self._store_id = id(self)
        self._initialized = False
        self._lock = asyncio.Lock()  # Add lock
        logger.info(f"Created SimpleShieldStore: {self._store_id}")

    async def initialize(self) -> None:
        """Initialize store if needed"""
        if self._initialized:
            return
        self._initialized = True
        logger.info(f"Shield store {self._store_id} initialized")

    async def register_detector_config(self, detector_id: str, config: Any) -> None:
        """Register detector configuration"""
        async with self._lock:
            self._detector_configs[detector_id] = config
            logger.info(
                f"Shield store {self._store_id} registered config for: {detector_id}"
            )

    async def get_shield(self, identifier: str) -> Shield:
        """Get or create shield by identifier"""
        await self.initialize()

        if identifier in self._shields:
            logger.debug(
                f"Shield store {self._store_id} found existing shield: {identifier}"
            )
            return self._shields[identifier]

        config = self._detector_configs.get(identifier)
        if config:
            logger.info(
                f"Shield store {self._store_id} creating shield for {identifier} using config"
            )

            # Extract detector params safely
            detector_params = {}
            if hasattr(config, "detector_params") and config.detector_params:
                if hasattr(config.detector_params, "__dict__"):
                    detector_params = {
                        k: v
                        for k, v in vars(config.detector_params).items()
                        if v is not None
                    }
                else:
                    # Handle case where detector_params is a dict or other type
                    detector_params = config.detector_params

            # Create shield with all required fields
            shield = Shield(
                identifier=identifier,
                provider_id="fms-safety",
                provider_resource_id=identifier,
                type="shield",
                name=f"{identifier} Shield",
                description=f"Safety shield for {identifier}",
                params=detector_params,  # Use extracted params
                metadata={
                    "detector_type": "content" if not config.is_chat else "chat",
                    "message_types": list(config.message_types),
                    "confidence_threshold": config.confidence_threshold,
                },
            )
            logger.info(
                f"Shield store {self._store_id} created shield: {identifier} with params: {detector_params}"
            )
        else:
            # Create basic shield if no config available
            shield = Shield(
                identifier=identifier,
                provider_id="fms-safety",
                provider_resource_id=identifier,
                type="shield",
                name=f"{identifier} Shield",
                description="Basic safety shield",
            )

            logger.info(
                f"Shield store {self._store_id} created basic shield: {identifier}"
            )

        self._shields[identifier] = shield
        return shield

    async def list_shields(self) -> ListShieldsResponse:
        """List all registered shields"""
        await self.initialize()
        shields = list(self._shields.values())
        shield_ids = [s.identifier for s in shields]
        logger.info(
            f"Shield store {self._store_id} listing {len(shields)} shields: {shield_ids}"
        )
        return ListShieldsResponse(data=shields)


class DetectorProvider(Safety, Shields):
    """Provider for managing safety detectors and shields"""

    def __init__(self, detectors: Dict[str, BaseDetector]) -> None:
        self.detectors = detectors
        self._shield_store = SimpleShieldStore()
        self._shields: Dict[str, Shield] = {}
        self._initialized = False
        self._provider_id = id(self)
        self._detector_key_to_id = {}  # Add mapping dict
        logger.info(
            f"Created DetectorProvider {self._provider_id} with {len(detectors)} detectors"
        )

        # Pre-register detector configs and map keys to IDs
        for detector_key, detector in detectors.items():
            detector.shield_store = self._shield_store
            config_id = detector.config.detector_id
            self._detector_key_to_id[detector_key] = config_id  # Store mapping
            self._shield_store.register_detector_config(config_id, detector.config)
            logger.info(
                f"Provider {self._provider_id} registered config for detector: {config_id}"
            )

    @property
    def shield_store(self) -> ShieldStore:
        return self._shield_store

    @shield_store.setter
    def shield_store(self, value: ShieldStore) -> None:
        if not value:
            logger.warning(f"Provider {self._provider_id} received null shield store")
            return

        logger.info(
            f"Provider {self._provider_id} setting new shield store: {id(value)}"
        )
        self._shield_store = value

        # Update detectors and sync shields
        for detector_id, detector in self.detectors.items():
            detector.shield_store = value
            logger.debug(
                f"Provider {self._provider_id} updated detector {detector_id} with shield store {id(value)}"
            )

    async def initialize(self) -> None:
        """Initialize provider and register initial shields"""
        if self._initialized:
            logger.info(f"Provider {self._provider_id} already initialized")
            return

        logger.info(f"Provider {self._provider_id} starting initialization")

        try:
            # Phase 1: Initialize detectors
            for detector_key, detector in self.detectors.items():
                await detector.initialize()

            # Phase 2: Create and register shields
            for detector_key, detector in self.detectors.items():
                try:
                    config_id = detector.config.detector_id  # Use config's detector_id
                    logger.info(
                        f"Provider {self._provider_id} creating shield for detector: {config_id}"
                    )
                    shield = await self._shield_store.get_shield(config_id)
                    if not shield:
                        raise DetectorValidationError(
                            f"Failed to create shield for detector: {config_id}"
                        )

                    # Register shield in both provider and store
                    self._shields[config_id] = shield
                    await detector.register_shield(shield)

                    # Ensure shield is in store's registry
                    store_shields = await self._shield_store.list_shields()
                    if shield not in store_shields.data:
                        self._shield_store._shields[config_id] = shield

                    logger.info(
                        f"Provider {self._provider_id} registered shield for detector: {config_id}"
                    )

                except Exception as e:
                    logger.error(
                        f"Provider {self._provider_id} failed to setup shield for detector {config_id}: {e}"
                    )
                    raise

            self._initialized = True
            shield_count = len(self._shields)
            shield_ids = list(self._shields.keys())
            logger.info(
                f"Provider {self._provider_id} initialization complete with {shield_count} shields: {shield_ids}"
            )

        except Exception as e:
            logger.error(f"Provider {self._provider_id} initialization failed: {e}")
            raise

    async def list_shields(self) -> ListShieldsResponse:
        """List all registered shields"""
        if not self._initialized:
            return await self.initialize()

        shields = list(self._shields.values())
        shield_ids = [s.identifier for s in shields]
        logger.info(
            f"Provider {self._provider_id} listing {len(shields)} shields: {shield_ids}"
        )
        return ListShieldsResponse(data=shields)

    async def get_shield(self, identifier: str) -> Optional[Shield]:
        """Get shield by identifier"""
        await self.initialize()

        # Return existing shield
        if identifier in self._shields:
            return self._shields[identifier]

        # Get detector and config
        detector = self.detectors.get(identifier)
        if not detector:
            return None

        # Create shield from store
        shield = await self._shield_store.get_shield(identifier)
        if shield:
            self._shields[identifier] = shield
            return shield

        return None

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Shield:
        """Register a new shield"""
        if not self._initialized:
            await self.initialize()

        # Return existing shield if already registered
        if shield_id in self._shields:
            return self._shields[shield_id]

        # Create new shield
        shield = await self._shield_store.get_shield(shield_id)
        if not shield:
            raise DetectorValidationError(f"Failed to create shield: {shield_id}")

        # Update fields if provided
        if provider_id:
            shield.provider_id = provider_id
        if provider_shield_id:
            shield.provider_resource_id = provider_shield_id
        if params is not None:
            shield.params = params

        # Register shield
        self._shields[shield_id] = shield

        # Register with detectors
        for detector in self.detectors.values():
            await detector.register_shield(shield)

        return shield

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Run shield against messages"""
        if not self._initialized:
            await self.initialize()

        if not messages:
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.INFO,
                    user_message="No messages to process",
                    metadata={"status": "skipped", "shield_id": shield_id},
                )
            )

        # Get all detectors for this shield
        shield_detectors = [
            detector
            for detector in self.detectors.values()
            if detector.config.detector_id == shield_id
        ]

        if not shield_detectors:
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=f"No detectors found for shield: {shield_id}",
                    metadata={
                        "status": "error",
                        "error_type": "detectors_not_found",
                        "shield_id": shield_id,
                    },
                )
            )

        try:
            detector_results = []
            has_violation = False

            # Run all detectors and collect results
            for detector in shield_detectors:
                filtered_messages = detector._filter_messages(messages)
                if not filtered_messages:
                    continue

                response = await detector.run_shield(
                    shield_id, filtered_messages, params
                )

                result = {
                    "detector_type": detector.__class__.__name__,
                    "threshold": detector.score_threshold,
                }

                if response.violation:
                    result.update(
                        {
                            "score": response.violation.metadata.get("score"),
                            "detection_type": response.violation.metadata.get(
                                "detection_type"
                            ),
                            "text": response.violation.metadata.get("text", ""),
                            "start": response.violation.metadata.get("start", 0),
                            "end": response.violation.metadata.get("end", 0),
                        }
                    )

                    if response.violation.violation_level == ViolationLevel.ERROR:
                        has_violation = True
                        result["status"] = "violation"
                    else:
                        result["status"] = "pass"
                else:
                    result["status"] = "pass"

                detector_results.append(result)

            # Construct final response
            if has_violation:
                return RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.ERROR,
                        user_message="Content failed safety checks",
                        metadata={
                            "status": "violation",
                            "shield_id": shield_id,
                            "detectors": detector_results,
                        },
                    )
                )
            else:
                return RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.INFO,
                        user_message="Content passed safety checks",
                        metadata={
                            "status": "pass",
                            "shield_id": shield_id,
                            "detectors": detector_results,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Provider {self._provider_id} shield execution failed: {e}")
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=f"Shield execution error: {str(e)}",
                    metadata={
                        "status": "error",
                        "error_type": "execution_error",
                        "shield_id": shield_id,
                        "error": str(e),
                    },
                )
            )

    async def shutdown(self) -> None:
        """Cleanup resources"""
        logger.info(f"Provider {self._provider_id} shutting down")
        errors = []

        for detector_id, detector in self.detectors.items():
            try:
                await detector.shutdown()
                logger.debug(
                    f"Provider {self._provider_id} shutdown detector: {detector_id}"
                )
            except Exception as e:
                error_msg = f"Error shutting down detector {detector_id}: {e}"
                logger.error(f"Provider {self._provider_id} {error_msg}")
                errors.append(error_msg)

        if errors:
            raise DetectorError(
                f"Provider {self._provider_id} shutdown errors: {', '.join(errors)}"
            )

        logger.info(f"Provider {self._provider_id} shutdown complete")
