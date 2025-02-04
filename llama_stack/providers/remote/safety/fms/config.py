from dataclasses import dataclass
from typing import Optional, List
from llama_models.schema_utils import json_schema_type


@json_schema_type
@dataclass
class FMSModelConfig:
    """Configuration for FMS safety model provider"""

    base_url: str
    detector_id: str
    confidence_threshold: float = 0.5
