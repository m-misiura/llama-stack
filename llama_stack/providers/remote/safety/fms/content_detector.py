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

    async def initialize(self) -> None:
        logger.info("Initializing FMS Model Adapter")
        pass

    async def shutdown(self) -> None:
        logger.info("Shutting down FMS Model Adapter")
        pass

    async def register_shield(self, shield: Shield) -> None:
        self.registered_shields.append(shield)

    async def _call_orchestrator_api(self, content: str) -> Dict:
        request = {"detectors": {self.config.detector_id: {}}, "content": content}

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

            return response.json()

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
                # Use FMS Orchestrator API: https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Orchestrator+API#/Task%20-%20Detection/api_v2_detection_text_content_unary_handler
                for content in contents:
                    result = await self._call_orchestrator_api(content)

                    for detection in result.get("detections", []):
                        if detection.get("score", 0) > self.config.confidence_threshold:
                            return RunShieldResponse(
                                violation=SafetyViolation(
                                    user_message=f"Content flagged as {detection['detection_type']} with confidence {detection['score']:.2f}",
                                    violation_level=ViolationLevel.ERROR,
                                    metadata={
                                        "detection_type": detection["detection_type"],
                                        "score": detection["score"],
                                        "detector_id": detection["detector_id"],
                                        "text": detection["text"],
                                        "start": detection["start"],
                                        "end": detection["end"],
                                    },
                                )
                            )
            else:
                # Use Detectors API: https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Detector+API#/Text/text_content_analysis_unary_handler
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
