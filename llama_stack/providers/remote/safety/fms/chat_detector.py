import logging
from typing import Any, Dict, List
import httpx

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from .config import FMSChatAdapterConfig

logger = logging.getLogger(__name__)


class FMSChatAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: FMSChatAdapterConfig) -> None:
        self.config = config
        self.registered_shields = []
        self.score_threshold = 0.5

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

        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        request = {
            "messages": chat_messages,
            "detector_params": {
                "temperature": self.config.temperature,
                **(
                    {"risk_name": self.config.risk_name}
                    if self.config.risk_name
                    else {}
                ),
                **(
                    {"risk_definition": self.config.risk_definition}
                    if self.config.risk_definition
                    else {}
                ),
                **(self.config.detector_params or {}),
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.base_url}/api/v1/text/chat",
                json=request,
                headers={
                    "detector-id": self.config.detector_id,
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"Error from Chat API: {response.text}")

            detections = response.json()

            if isinstance(detections, list) and len(detections) > 0:
                detection = detections[0]
                if (
                    detection.get("detection") == "Yes"
                    and detection.get("score", 0) > self.score_threshold
                ):
                    return RunShieldResponse(
                        violation=SafetyViolation(
                            user_message=f"Content violation detected ({detection.get('detection_type')}) with confidence {detection.get('score'):.2f}",
                            violation_level=ViolationLevel.ERROR,
                            metadata={
                                "detection_type": detection.get("detection_type"),
                                "score": detection.get("score"),
                                "risk_name": self.config.risk_name,
                                "risk_definition": self.config.risk_definition,
                            },
                        )
                    )

        return RunShieldResponse()
