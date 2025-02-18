from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional, TypeVar, Type
from dataclasses import dataclass

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse, SafetyViolation, ViolationLevel
from llama_stack.providers.remote.safety.fms.detectors.base import (
    BaseDetector,
    DetectorError,
    DetectorRequestError,
    DetectorValidationError,
    DetectionResult,
)
from llama_stack.providers.remote.safety.fms.config import ChatDetectorConfig

# Type aliases for better readability
ChatMessage = Dict[str, str]
ChatRequest = Dict[str, Any]
DetectorResponse = List[Dict[str, Any]]

logger = logging.getLogger(__name__)


class ChatDetectorError(DetectorError):
    """Specific errors for chat detector operations"""

    pass


@dataclass(frozen=True)
class ChatDetectionMetadata:
    """Structured metadata for chat detections"""

    risk_name: Optional[str] = None
    risk_definition: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format"""
        result = {}
        if self.risk_name:
            result["risk_name"] = self.risk_name
        if self.risk_definition:
            result["risk_definition"] = self.risk_definition
        if self.additional_metadata:
            result["metadata"] = self.additional_metadata
        return result


class ChatDetector(BaseDetector):
    """Detector for chat-based safety checks"""

    def __init__(self, config: ChatDetectorConfig) -> None:
        """Initialize chat detector with configuration"""
        if not isinstance(config, ChatDetectorConfig):
            raise DetectorValidationError(
                "Config must be an instance of ChatDetectorConfig"
            )
        super().__init__(config)
        self.config: ChatDetectorConfig = config
        logger.info(f"Initialized ChatDetector with config: {vars(config)}")

    def _prepare_chat_request(
        self, messages: List[ChatMessage], params: Optional[Dict[str, Any]] = None
    ) -> ChatRequest:
        """Prepare the request based on API mode"""
        detector_params = self._extract_detector_params()

        if self.config.use_orchestrator_api:
            return {
                "detectors": {self.config.detector_id: detector_params},
                "messages": messages,
            }

        return {
            "messages": messages,
            "detector_params": detector_params if detector_params else params or {},
        }

    def _extract_detector_params(self) -> Dict[str, Any]:
        """Extract non-null detector parameters"""
        if not self.config.detector_params:
            return {}

        params = {
            k: v for k, v in vars(self.config.detector_params).items() if v is not None
        }
        logger.debug(f"Extracted detector params: {params}")
        return params

    async def _call_detector_api(
        self,
        messages: List[ChatMessage],
        params: Optional[Dict[str, Any]] = None,
    ) -> DetectorResponse:
        """Call chat detector API with proper endpoint selection"""
        try:
            request = self._prepare_chat_request(messages, params)
            headers = self._prepare_headers()

            logger.info("Making detector API request")
            logger.debug(f"Request headers: {headers}")
            logger.debug(f"Request payload: {request}")

            response = await self._make_request(request, headers)
            return self._extract_detections(response)

        except Exception as e:
            logger.error(f"API call failed: {str(e)}", exc_info=True)
            raise DetectorRequestError(
                f"Chat detector API call failed: {str(e)}"
            ) from e

    def _extract_detections(self, response: Dict[str, Any]) -> DetectorResponse:
        """Extract detections from API response"""
        if self.config.use_orchestrator_api:
            detections = response.get("detections", [])
            logger.debug(f"Orchestrator detections: {detections}")
            return detections

        # Direct API returns a list where first item contains detections
        if isinstance(response, list) and response:
            detections = (
                [response[0]] if not isinstance(response[0], list) else response[0]
            )
            logger.debug(f"Direct API detections: {detections}")
            return detections

        logger.debug("No detections found in response")
        return []

    def _process_detection(
        self, detection: Dict[str, Any]
    ) -> Optional[DetectionResult]:
        """Process detection result and validate against threshold"""
        if not detection.get("score"):
            logger.warning("Detection missing score field")
            return None

        score = detection.get("score", 0)
        if score > self.score_threshold:
            metadata = ChatDetectionMetadata(
                risk_name=(
                    self.config.detector_params.risk_name
                    if self.config.detector_params
                    else None
                ),
                risk_definition=(
                    self.config.detector_params.risk_definition
                    if self.config.detector_params
                    else None
                ),
                additional_metadata=detection.get("metadata"),
            )

            return DetectionResult(
                detection="Yes",
                detection_type=detection["detection_type"],
                score=score,
                detector_id=detection.get("detector_id", self.config.detector_id),
                text=detection.get("text", ""),
                start=detection.get("start", 0),
                end=detection.get("end", 0),
                metadata=metadata.to_dict(),
            )
        return None

    async def _run_shield_impl(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Implementation of shield checks for chat messages"""
        try:
            shield = await self.shield_store.get_shield(shield_id)
            self._validate_shield(shield)

            logger.info(f"Processing {len(messages)} message(s)")
            chat_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            detections = await self._call_detector_api(chat_messages, params)

            for detection in detections:
                if processed := self._process_detection(detection):
                    logger.info(f"Violation detected: {processed}")
                    return self.create_violation_response(
                        processed, detection.get("detector_id", self.config.detector_id)
                    )

            logger.debug("No violations detected")
            return RunShieldResponse()

        except Exception as e:
            logger.error(f"Shield execution failed: {str(e)}", exc_info=True)
            raise ChatDetectorError(f"Shield execution failed: {str(e)}") from e
