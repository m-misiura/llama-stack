import logging
from typing import Dict, List, Any, Optional
from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse, SafetyViolation, ViolationLevel
from .base_detector import BaseDetector
from .config import ChatDetectorConfig

logger = logging.getLogger(__name__)


class ChatDetector(BaseDetector):
    """Detector for chat-based safety checks"""

    def __init__(self, config: ChatDetectorConfig) -> None:
        super().__init__(config)
        self.config: ChatDetectorConfig = config

    async def _call_detector_api(
        self,
        messages: List[Dict[str, str]],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Call chat detector API"""
        if self.config.use_orchestrator_api:
            detector_config = self.config.guardrails_detectors.get("granite", {})
            if self.config.risk_name:
                detector_config.update(
                    {
                        "risk_name": self.config.risk_name,
                        "risk_definition": self.config.risk_definition,
                    }
                )

            request = {"detectors": {"granite": detector_config}, "messages": messages}
            response = await self._make_request(
                f"{self.config.base_url}/api/v2/text/detection/chat",
                request,
            )
            if response and response.get("detections"):
                detection = response["detections"][0]
                # Add risk parameters from detector config to detection
                detection.update(detector_config)
                return detection
            return {}
        else:
            request = {
                "messages": messages,
                "detector_params": {
                    "temperature": self.config.temperature,
                    "risk_name": self.config.risk_name,  # Always include risk params
                    "risk_definition": self.config.risk_definition,
                    **(self.config.detector_params or {}),
                },
            }
            headers = {"detector-id": self.config.detector_id}
            response = await self._make_request(
                f"{self.config.base_url}/api/v1/text/chat", request, headers
            )
            result = response[0] if isinstance(response, list) and response else {}
            # Add risk parameters to result for consistency
            if self.config.risk_name:
                result["risk_name"] = self.config.risk_name
                result["risk_definition"] = self.config.risk_definition
            return result

    def create_violation_response(
        self, detection: Dict, detector_id: str
    ) -> RunShieldResponse:
        """Create standardized violation response"""
        metadata = {
            "detection_type": detection.get("detection_type", "risk"),
            "score": detection.get("score", 0),
            "detector_id": detector_id,
            "text": detection.get("text", ""),
            "start": detection.get("start", 0),
            "end": detection.get("end", 0),
        }

        # Add risk parameters from either detection or config
        if "risk_name" in detection:
            metadata["risk_name"] = detection["risk_name"]
            metadata["risk_definition"] = detection.get("risk_definition")
        elif self.config.risk_name:
            metadata["risk_name"] = self.config.risk_name
            metadata["risk_definition"] = self.config.risk_definition

        return RunShieldResponse(
            violation=SafetyViolation(
                user_message=f"Content flagged by {detector_id} as {metadata['detection_type']} with confidence {detection['score']:.2f}",
                violation_level=ViolationLevel.ERROR,
                metadata=metadata,
            )
        )

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

            chat_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            result = await self._call_detector_api(chat_messages, params)
            if (
                result.get("detection") == "Yes"
                and result.get("score", 0) > self.score_threshold
            ):
                return self.create_violation_response(result, self.config.detector_id)

            return RunShieldResponse()

        except Exception as e:
            logger.error(f"Error in run_shield: {str(e)}")
            raise
