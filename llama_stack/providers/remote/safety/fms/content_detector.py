import logging
from typing import Any, Dict, List

import httpx
from llama_stack.apis.inference import Message, SystemMessage, UserMessage
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate

from .config import FMSModelConfig

logger = logging.getLogger(__name__)


class FMSModelAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: FMSModelConfig) -> None:
        self.config = config
        self.registered_shields = []
        self.score_threshold = 0.5

    async def initialize(self) -> None:
        logger.info("Initializing FMS Model Adapter")
        pass

    async def shutdown(self) -> None:
        logger.info("Shutting down FMS Model Adapter")
        pass

    async def register_shield(self, shield: Shield) -> None:
        self.registered_shields.append(shield)

    async def _call_orchestrator_api(self, content: str) -> Dict:
        detectors = self.config.guardrails_detectors or {self.config.detector_id: {}}

        request = {"detectors": detectors, "content": content}

        logger.debug(f"Calling orchestrator API with request: {request}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.base_url}/api/v2/text/detection/content",
                json=request,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"Error from Guardrails API: {response.text}")

            result = response.json()
            logger.debug(f"Orchestrator API response: {result}")
            return result

    async def _call_detectors_api(self, contents: List[str]) -> Dict:
        request = {"contents": contents}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.base_url}/api/v1/text/contents",
                json=request,
                headers={"detector-id": self.config.detector_id},
            )

            if response.status_code != 200:
                raise RuntimeError(f"Error from Content API: {response.text}")

            return response.json()

    def _get_detector_violation(
        self, detections: List[Dict], expected_detector: str = None
    ) -> Dict:
        # Filter by threshold
        valid_detections = [
            d
            for d in detections
            if d.get("score", 0) > self.config.confidence_threshold
        ]

        if not valid_detections:
            return None

        # If expected detector specified, check its detections first
        if expected_detector:
            detector_matches = [
                d for d in valid_detections if d.get("detector_id") == expected_detector
            ]
            if detector_matches:
                return max(detector_matches, key=lambda x: x.get("score", 0))

        # No match for expected detector, return highest scoring detection
        return max(valid_detections, key=lambda x: x.get("score", 0))

    async def run_shield(
        self, shield_id: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        try:
            shield = await self.shield_store.get_shield(shield_id)
            if not shield:
                raise ValueError(f"Shield {shield_id} not found")

            contents = []
            for msg in messages:
                if isinstance(msg, (UserMessage, SystemMessage)):
                    contents.append(msg.content)

            if not contents:
                return RunShieldResponse()

            if self.config.use_orchestrator_api:
                for content in contents:
                    result = await self._call_orchestrator_api(content)
                    expected_detector = (
                        params.get("expected_detector")
                        if params
                        else shield.provider_resource_id
                    )

                    violation_detection = self._get_detector_violation(
                        result.get("detections", []), expected_detector
                    )

                    if violation_detection:
                        return RunShieldResponse(
                            violation=SafetyViolation(
                                user_message=f"Content flagged by {violation_detection['detector_id']} as {violation_detection['detection_type']} with confidence {violation_detection['score']:.2f}",
                                violation_level=ViolationLevel.ERROR,
                                metadata={
                                    "detection_type": violation_detection[
                                        "detection_type"
                                    ],
                                    "score": violation_detection["score"],
                                    "detector_id": violation_detection["detector_id"],
                                    "text": violation_detection["text"],
                                    "start": violation_detection["start"],
                                    "end": violation_detection["end"],
                                },
                            )
                        )
            else:
                result = await self._call_detectors_api(contents)
                for analysis_list in result:
                    for analysis in analysis_list:
                        if analysis["score"] > self.config.confidence_threshold:
                            return RunShieldResponse(
                                violation=SafetyViolation(
                                    user_message=f"Content flagged as {analysis['sequence_classification']} with confidence {analysis['score']:.2f}",
                                    violation_level=ViolationLevel.ERROR,
                                    metadata={
                                        "detection_type": analysis["detection_type"],
                                        "score": analysis["score"],
                                    },
                                )
                            )

            return RunShieldResponse()

        except Exception as e:
            logger.error(f"Error in run_shield: {str(e)}")
            raise
