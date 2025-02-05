from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from llama_models.schema_utils import json_schema_type


@json_schema_type
@dataclass
class FMSModelConfig:
    """Configuration for FMS safety model provider"""

    base_url: str
    detector_id: str
    confidence_threshold: float = 0.5
    allow_list: Optional[List[str]] = None
    block_list: Optional[List[str]] = None
    use_orchestrator_api: bool = False
    guardrails_detectors: Optional[Dict[str, Dict]] = None


@json_schema_type
@dataclass
class FMSChatAdapterConfig:
    """Configuration for FMS safety model provider"""

    base_url: str
    detector_id: str
    temperature: float = 0.0
    risk_name: Optional[str] = None
    risk_definition: Optional[str] = None
    detector_params: Optional[Dict[str, Any]] = None
    use_orchestrator_api: bool = False
    guardrails_detectors: Optional[Dict[str, Dict]] = None
