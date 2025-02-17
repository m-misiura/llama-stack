import logging
from typing import Dict, List, Any, Optional
import httpx
from abc import ABC, abstractmethod
from llama_stack.apis.inference import (
    Message,
    UserMessage,
    SystemMessage,
    ToolResponseMessage,
    CompletionMessage,
)
from llama_stack.apis.safety import (
    Safety,
    RunShieldResponse,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.remote.safety.fms_refactor3.config import (
    BaseDetectorConfig,
    DetectorParams,
    EndpointType,
)

logger = logging.getLogger(__name__)


class BaseDetector(Safety, ShieldsProtocolPrivate, ABC):
    """Base class for all safety detectors"""

    def __init__(self, config: BaseDetectorConfig) -> None:
        self.config = config
        self.registered_shields = []
        self.score_threshold = config.confidence_threshold

    async def initialize(self) -> None:
        """Initialize detector resources"""
        logger.info(f"Initializing {self.__class__.__name__}")

    async def shutdown(self) -> None:
        """Clean up detector resources"""
        logger.info(f"Shutting down {self.__class__.__name__}")

    async def register_shield(self, shield: Shield) -> None:
        """Register a shield with the detector"""
        logger.info(f"Registering shield {shield.identifier}")
        self.registered_shields.append(shield)

    def _should_process_message(self, message: Message) -> bool:
        """Check if this detector should process the given message type"""
        message_type_map = {
            "user": lambda m: isinstance(m, UserMessage),
            "system": lambda m: isinstance(m, SystemMessage),
            "tool": lambda m: isinstance(m, ToolResponseMessage),
            "completion": lambda m: isinstance(m, CompletionMessage),
        }

        for message_type in self.config.message_types:
            if message_type_map.get(message_type, lambda _: False)(message):
                return True
        return False

    def _filter_messages(self, messages: List[Message]) -> List[Message]:
        """Filter messages based on configured message types"""
        return [msg for msg in messages if self._should_process_message(msg)]

    def _construct_url(self) -> str:
        """Construct API URL based on configuration"""
        if self.config.use_orchestrator_api:
            if not self.config.orchestrator_base_url:
                raise ValueError(
                    "orchestrator_base_url is required when use_orchestrator_api is True"
                )
            base_url = self.config.orchestrator_base_url
            endpoint_info = (
                EndpointType.ORCHESTRATOR_CHAT.value
                if self.config.is_chat
                else EndpointType.ORCHESTRATOR_CONTENT.value
            )
            endpoint = endpoint_info["path"]
        else:
            if not self.config.base_url:
                raise ValueError(
                    "base_url is required when use_orchestrator_api is False"
                )
            base_url = self.config.base_url
            endpoint_info = (
                EndpointType.DIRECT_CHAT.value
                if self.config.is_chat
                else EndpointType.DIRECT_CONTENT.value
            )
            endpoint = endpoint_info["path"]

        url = f"{base_url.rstrip('/')}{endpoint}"
        logger.debug(
            f"Constructed URL: {url} for {'chat' if self.config.is_chat else 'content'} endpoint"
        )
        return url

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare request headers based on configuration"""
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        if not self.config.use_orchestrator_api and self.config.detector_id:
            headers["detector-id"] = self.config.detector_id

        return headers

    def _prepare_request_payload(
        self, messages: List[Message], params: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Prepare request payload based on endpoint type and orchestrator mode"""
        detector_params = {}
        if self.config.detector_params:
            detector_params = {
                k: v
                for k, v in vars(self.config.detector_params).items()
                if v is not None
            }

        if self.config.use_orchestrator_api:
            payload = {"detectors": {self.config.detector_id: detector_params}}
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

        logger.debug(
            f"Prepared payload for {'chat' if self.config.is_chat else 'content'} endpoint: {payload}"
        )
        return payload

    async def _make_request(
        self,
        request: Dict,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """Make HTTP request with error handling"""
        url = self._construct_url()
        default_headers = self._prepare_headers()

        if headers:
            default_headers.update(headers)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, json=request, headers=default_headers, timeout=timeout
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.text}")
            raise RuntimeError(f"API Error: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {str(e)}")
            raise RuntimeError(f"Request Error: {str(e)}")

    def _process_detection(self, detection: Dict) -> Optional[Dict]:
        """Process detection result and validate against threshold"""
        if not detection.get("score"):
            logger.warning("Detection missing score field")
            return None

        if detection.get("score", 0) > self.score_threshold:
            return {
                "detection": "Yes",
                "detection_type": detection["detection_type"],
                "score": detection["score"],
                "detector_id": detection.get("detector_id", self.config.detector_id),
                "text": detection.get("text", ""),
                "start": detection.get("start", 0),
                "end": detection.get("end", 0),
                **(
                    {"metadata": detection["metadata"]}
                    if "metadata" in detection
                    else {}
                ),
            }
        return None

    def create_violation_response(
        self,
        detection: Dict,
        detector_id: str,
        level: ViolationLevel = ViolationLevel.ERROR,
    ) -> RunShieldResponse:
        """Create standardized violation response"""
        return RunShieldResponse(
            violation=SafetyViolation(
                user_message=f"Content flagged by {detector_id} as {detection['detection_type']} with confidence {detection['score']:.2f}",
                violation_level=level,
                metadata={
                    "detection_type": detection["detection_type"],
                    "score": detection["score"],
                    "detector_id": detector_id,
                    "text": detection.get("text", ""),
                    "start": detection.get("start", 0),
                    "end": detection.get("end", 0),
                    **(
                        {"metadata": detection["metadata"]}
                        if "metadata" in detection
                        else {}
                    ),
                    **(
                        {"risk_name": detection.get("risk_name")}
                        if detection.get("risk_name")
                        else {}
                    ),
                    **(
                        {"risk_definition": detection.get("risk_definition")}
                        if detection.get("risk_definition")
                        else {}
                    ),
                },
            )
        )

    def _validate_shield(self, shield: Shield) -> None:
        """Validate shield configuration"""
        if not shield:
            raise ValueError("Shield not found")
        if not shield.identifier:
            raise ValueError("Shield missing identifier")

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
        # Filter messages based on configured message types
        filtered_messages = self._filter_messages(messages)

        if not filtered_messages:
            logger.debug(
                f"No messages of configured types {self.config.message_types} to process"
            )
            return RunShieldResponse(violation=None)

        # Continue with shield processing using filtered messages
        return await self._run_shield_impl(shield_id, filtered_messages, params)


class DetectorProvider:
    """Provider that manages multiple detectors and allows running them all at once"""

    def __init__(self, detectors: Dict[str, BaseDetector]):
        self.detectors = detectors

    async def run_shield(
        self, shield_id: str, messages: List[Message]
    ) -> Dict[str, RunShieldResponse]:
        """Run all detectors and return results from each"""
        results = {}
        for name, detector in self.detectors.items():
            response = await detector.run_shield(shield_id, messages)
            results[name] = response
        return results

    async def shutdown(self):
        """Shutdown all detectors"""
        for detector in self.detectors.values():
            await detector.shutdown()
