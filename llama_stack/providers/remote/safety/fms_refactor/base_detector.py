import logging
from typing import Dict, List, Any, Optional
import httpx
from abc import ABC, abstractmethod
from llama_stack.apis.inference import Message
from llama_stack.apis.safety import (
    Safety,
    RunShieldResponse,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from .config import BaseDetectorConfig

logger = logging.getLogger(__name__)


class BaseDetector(Safety, ShieldsProtocolPrivate, ABC):
    """Base class for all safety detectors"""

    def __init__(self, config: BaseDetectorConfig) -> None:
        self.config = config
        self.registered_shields = []
        self.score_threshold = 0.5

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

    async def _make_request(
        self,
        url: str,
        request: Dict,
        headers: Optional[Dict] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """Make HTTP request with error handling"""
        default_headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
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
                "detector_id": detection.get("detector_id"),
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
    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Run safety checks using configured shield"""
        pass
