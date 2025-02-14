import logging
from typing import Dict, List, Any, Optional
import asyncio
from llama_stack.apis.inference import Message, SystemMessage, UserMessage
from llama_stack.apis.safety import RunShieldResponse
from .base_detector import BaseDetector
from .config import ContentDetectorConfig, BaseDetectorConfig

logger = logging.getLogger(__name__)


class ContentDetector(BaseDetector):
    """Detector for content-based safety checks"""

    def __init__(self, config: ContentDetectorConfig) -> None:
        super().__init__(config)
        self.config: ContentDetectorConfig = config

    def _process_detection(self, detection_list: List[Dict]) -> Optional[Dict]:
        """Process detection result and validate against threshold"""
        if not isinstance(detection_list, list):
            logger.warning(f"Expected list, got {type(detection_list)}")
            return None

        for detection in detection_list:
            if not isinstance(detection, dict):
                continue

            score = detection.get("score", 0)
            if score > self.score_threshold:
                return {
                    "detection": "Yes",
                    "detection_type": detection.get("detection_type", "unknown"),
                    "score": score,
                    "detector_id": detection.get("detector_id"),
                    "text": detection.get("text", ""),
                    "start": detection.get("start", 0),
                    "end": detection.get("end", 0),
                }
        return None

    async def _call_detector_api(
        self,
        content: str,
        detector: BaseDetectorConfig,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Call detector API with proper endpoint selection"""
        if self.config.use_orchestrator_api:
            request = {
                "detectors": self.config.guardrails_detectors,
                "content": content,  # Changed from list to string
            }
            response = await self._make_request(
                f"{detector.base_url}/api/v2/text/detection/content", request
            )
            return response.get("detections", []) if response else []
        else:
            # Original individual detector logic
            request = {
                "contents": [content],
                "detector_params": detector.detector_params or {},
            }
            headers = {"detector-id": detector.detector_id}
            response = await self._make_request(
                f"{detector.base_url}/api/v1/text/contents", request, headers
            )
            return response[0] if response else []

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Execute shield checks"""
        try:
            shield = await self.shield_store.get_shield(shield_id)
            if not shield:
                raise ValueError(f"Shield {shield_id} not found")

            contents = [
                msg.content
                for msg in messages
                if isinstance(msg, (UserMessage, SystemMessage))
            ]

            if not contents:
                return RunShieldResponse()

            for content in contents:
                if self.config.use_orchestrator_api:
                    # Call orchestrator API and process all detections
                    result = await self._call_detector_api(content, self.config, params)
                    for detection in result:
                        if detection.get("score", 0) > self.score_threshold:
                            processed = {
                                "detection": "Yes",
                                "detection_type": detection.get(
                                    "detection_type", "unknown"
                                ),
                                "score": detection.get("score", 0),
                                "detector_id": detection.get("detector_id"),
                                "text": detection.get("text", ""),
                                "start": detection.get("start", 0),
                                "end": detection.get("end", 0),
                            }
                            return self.create_violation_response(
                                processed,
                                detection.get("detector_id", self.config.detector_id),
                            )
                else:
                    # Original individual detector logic
                    main_result = await self._call_detector_api(
                        content, self.config, params
                    )
                    if processed := self._process_detection(main_result):
                        return self.create_violation_response(
                            processed, self.config.detector_id
                        )

                    if self.config.additional_detectors:
                        for detector in self.config.additional_detectors:
                            result = await self._call_detector_api(
                                content, detector, params
                            )
                            if processed := self._process_detection(result):
                                return self.create_violation_response(
                                    processed, detector.detector_id
                                )

            return RunShieldResponse()

        except Exception as e:
            logger.error(f"Error in run_shield: {str(e)}")
            raise
