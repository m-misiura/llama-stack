from typing import Any, Union, Dict, List
from llama_stack.providers.remote.safety.fms.config import (
    ContentDetectorConfig,
    ChatDetectorConfig,
    FMSSafetyProviderConfig,
    EndpointType,
    DetectorParams,
)
from llama_stack.providers.remote.safety.fms.detectors.chat_detector import (
    ChatDetector,
)
from llama_stack.providers.remote.safety.fms.detectors.content_detector import (
    ContentDetector,
)
from llama_stack.providers.remote.safety.fms.detectors.base import (
    BaseDetector,
    DetectorProvider,
)


async def get_adapter_impl(
    config: Union[ContentDetectorConfig, ChatDetectorConfig, FMSSafetyProviderConfig],
    _deps: Any = None,
) -> Union[BaseDetector, DetectorProvider]:
    """Get appropriate detector implementation(s) based on config type"""

    # Handle provider config with multiple detectors
    if isinstance(config, FMSSafetyProviderConfig):
        detectors = {}
        # Process all detectors from the config
        for detector_id, detector_config in config.detectors.items():
            if isinstance(detector_config, (ChatDetectorConfig, ContentDetectorConfig)):
                # Config validation now happens in __post_init__
                impl = await get_adapter_impl(detector_config)
                detectors[detector_id] = impl
            else:
                raise ValueError(f"Invalid detector config type for {detector_id}")
        return DetectorProvider(detectors)

    # Handle single detector config (unchanged)
    if isinstance(config, ChatDetectorConfig):
        impl = ChatDetector(config)
    elif isinstance(config, ContentDetectorConfig):
        impl = ContentDetector(config)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")

    await impl.initialize()
    return impl


__all__ = [
    "get_adapter_impl",  # Main factory function
    "ContentDetectorConfig",  # Base configs for detectors
    "ChatDetectorConfig",
    "FMSSafetyProviderConfig",  # Main provider config
    "EndpointType",  # Endpoint type enum
    "DetectorParams",  # Parameters for detectors
    "ChatDetector",  # Detector implementations
    "ContentDetector",
    "BaseDetector",  # Base classes
    "DetectorProvider",  # Added this as it's used in the return type
]
