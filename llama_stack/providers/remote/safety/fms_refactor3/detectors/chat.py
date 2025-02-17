import logging
from typing import Dict, List, Any, Optional
from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse, SafetyViolation, ViolationLevel
from llama_stack.providers.remote.safety.fms_refactor3.detectors.base import (
    BaseDetector,
)
from llama_stack.providers.remote.safety.fms_refactor3.config import (
    ChatDetectorConfig,
)

logger = logging.getLogger(__name__)


class ChatDetector(BaseDetector):
    """Detector for chat-based safety checks"""

    def __init__(self, config: ChatDetectorConfig) -> None:
        if not isinstance(config, ChatDetectorConfig):
            raise ValueError("Config must be an instance of ChatDetectorConfig")
        super().__init__(config)
        self.config: ChatDetectorConfig = config
        logger.info(f"Initialized ChatDetector with config: {vars(config)}")

    def _prepare_chat_request(
        self, messages: List[Dict[str, str]], params: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Prepare the request based on API mode"""
        detector_params = {}
        if self.config.detector_params:
            detector_params = {
                k: v
                for k, v in vars(self.config.detector_params).items()
                if v is not None
            }
            logger.debug(f"Using detector params: {detector_params}")

        if self.config.use_orchestrator_api:
            request = {
                "detectors": {self.config.detector_id: detector_params},
                "messages": messages,
            }
        else:
            request = {
                "messages": messages,
                "detector_params": detector_params if detector_params else params or {},
            }

        logger.debug(f"Prepared request: {request}")
        return request

    async def _call_detector_api(
        self,
        messages: List[Dict[str, str]],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Call chat detector API with proper endpoint selection"""
        request = self._prepare_chat_request(messages, params)
        headers = self._prepare_headers()

        logger.info("Making detector API request")
        logger.debug(f"Request headers: {headers}")
        logger.debug(f"Request payload: {request}")

        try:
            response = await self._make_request(request, headers)
            logger.debug(f"Raw API response: {response}")

            if self.config.use_orchestrator_api:
                detections = response.get("detections", [])
                logger.debug(f"Orchestrator detections: {detections}")
                return detections
            else:
                # Direct API returns a list where first item contains detections
                if isinstance(response, list) and response:
                    detections = (
                        [response[0]]
                        if not isinstance(response[0], list)
                        else response[0]
                    )
                    logger.debug(f"Direct API detections: {detections}")
                    return detections
                logger.debug("No detections found in response")
                return []
        except Exception as e:
            logger.error(f"API call failed: {str(e)}", exc_info=True)
            raise

    def _process_detection(self, detection: Dict) -> Optional[Dict]:
        """Process detection result and validate against threshold"""
        if not detection.get("score"):
            logger.warning("Detection missing score field")
            return None

        if detection.get("score", 0) > self.score_threshold:
            result = {
                "detection": "Yes",
                "detection_type": detection["detection_type"],
                "score": detection["score"],
                "detector_id": detection.get("detector_id", self.config.detector_id),
                "text": detection.get("text", ""),
                "start": detection.get("start", 0),
                "end": detection.get("end", 0),
            }

            # Add risk-specific fields if present in detector params
            if self.config.detector_params:
                if self.config.detector_params.risk_name:
                    result["risk_name"] = self.config.detector_params.risk_name
                if self.config.detector_params.risk_definition:
                    result["risk_definition"] = (
                        self.config.detector_params.risk_definition
                    )

            # Add any additional metadata from the detection
            if "metadata" in detection:
                result["metadata"] = detection["metadata"]

            logger.debug(f"Processed detection result: {result}")
            return result
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
                processed = self._process_detection(detection)
                if processed:
                    logger.info(f"Violation detected: {processed}")
                    return self.create_violation_response(
                        processed, detection.get("detector_id", self.config.detector_id)
                    )

            logger.debug("No violations detected")
            return RunShieldResponse()

        except Exception as e:
            logger.error(f"Shield execution failed: {str(e)}", exc_info=True)
            raise
