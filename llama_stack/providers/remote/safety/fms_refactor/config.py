from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from llama_models.schema_utils import json_schema_type


@json_schema_type
@dataclass
class BaseDetectorConfig:
    """Base configuration for all detectors"""

    base_url: str
    detector_id: str
    confidence_threshold: float = 0.5
    detector_params: Optional[Dict[str, Any]] = None


@json_schema_type
@dataclass
class ContentDetectorConfig(BaseDetectorConfig):
    """Configuration for content detectors"""

    additional_detectors: Optional[List[BaseDetectorConfig]] = None
    allow_list: Optional[List[str]] = None
    block_list: Optional[List[str]] = None
    use_orchestrator_api: bool = False
    guardrails_detectors: Optional[Dict[str, Dict]] = None


@json_schema_type
@dataclass
class ChatDetectorConfig(BaseDetectorConfig):
    """Configuration for chat detectors"""

    temperature: float = 0.0
    risk_name: Optional[str] = None
    risk_definition: Optional[str] = None
    use_orchestrator_api: bool = False
    guardrails_detectors: Optional[Dict[str, Dict]] = None
