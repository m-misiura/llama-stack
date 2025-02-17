from typing import Any, Union, Dict, List
from .config import (
    ContentDetectorConfig,
    ChatDetectorConfig,
    FMSSafetyProviderConfig,
    EndpointType,
    DetectorParams,
)
from .chat_detector import ChatDetector
from .content_detector import ContentDetector
from .base_detector import BaseDetector, DetectorProvider


async def get_adapter_impl(
    config: Union[ContentDetectorConfig, ChatDetectorConfig, FMSSafetyProviderConfig],
    _deps: Any = None,
) -> Union[BaseDetector, DetectorProvider]:
    """Get appropriate detector implementation(s) based on config type"""

    # Handle provider config with multiple detectors
    if isinstance(config, FMSSafetyProviderConfig):
        detectors = {}
        for detector_id, detector_config in config.detectors.items():
            if isinstance(detector_config, (ChatDetectorConfig, ContentDetectorConfig)):
                # Validate detector config after settings are propagated
                detector_config.validate()
                impl = await get_adapter_impl(detector_config)
                detectors[detector_id] = impl
            else:
                raise ValueError(f"Invalid detector config type for {detector_id}")
        return DetectorProvider(detectors)

    # Handle single detector config
    if isinstance(config, ChatDetectorConfig):
        config.validate()  # Validate before creating implementation
        impl = ChatDetector(config)
    elif isinstance(config, ContentDetectorConfig):
        config.validate()  # Validate before creating implementation
        impl = ContentDetector(config)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")

    await impl.initialize()
    return impl


__all__ = [
    "get_adapter_impl",
    "ContentDetectorConfig",
    "ChatDetectorConfig",
    "FMSSafetyProviderConfig",
    "EndpointType",
    "DetectorParams",
    "ChatDetector",
    "ContentDetector",
    "BaseDetector",
]
