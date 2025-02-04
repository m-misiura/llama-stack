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
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        pass

    async def run_shield(
        self, shield_id: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")

        # Extract content from messages
        contents = []
        for msg in messages:
            if isinstance(msg, (UserMessage, SystemMessage)):
                contents.append(msg.content)

        request = {"contents": contents}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.base_url}/api/v1/text/contents",
                json=request,
                headers={"detector-id": self.config.detector_id},
            )

            if response.status_code != 200:
                raise RuntimeError(f"Error from FMS model: {response.text}")

            result = response.json()

            # Check for violations from model classifications
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
