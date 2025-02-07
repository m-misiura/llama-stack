from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from llama_models.schema_utils import json_schema_type


@json_schema_type
@dataclass
class BaseContentDetectionConfig:
    """Configuration for FMS safety model provider"""

    base_url: str
    detector_id: str
    confidence_threshold: float = 0.5


@json_schema_type
@dataclass
class FMSModelConfig:
    """Configuration for FMS safety model provider"""

    detectors: List[BaseContentDetectionConfig]
    confidence_threshold: float = 0.5  # Global threshold
    allow_list: Optional[List[str]] = None
    block_list: Optional[List[str]] = None
    use_orchestrator_api: bool = False
    guardrails_detectors: Optional[Dict[str, Dict]] = None

    def __post_init__(self):
        if self.use_orchestrator_api and not self.guardrails_detectors:
            # For orchestrator, construct detectors dict from detector list
            self.guardrails_detectors = {
                detector.detector_id: {} for detector in self.detectors
            }

    def get_detector(self, detector_id: str) -> Optional[BaseContentDetectionConfig]:
        """Get detector config by ID"""
        for detector in self.detectors:
            if detector.detector_id == detector_id:
                return detector
        return None


@json_schema_type
@dataclass
class ChatDetectionConfig:
    """Configuration for FMS safety model provider"""

    base_url: str
    detector_id: str
    temperature: float = 0.0
    risk_name: Optional[str] = None
    risk_definition: Optional[str] = None
    detector_params: Optional[Dict[str, Any]] = None
    use_orchestrator_api: bool = False
    guardrails_detectors: Optional[Dict[str, Dict]] = None
