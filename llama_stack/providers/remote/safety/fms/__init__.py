from typing import Any, Dict, Union

from llama_stack.providers.remote.safety.fms.config import (
    ChatDetectorConfig,
    ContentDetectorConfig,
    DetectorParams,
    EndpointType,
    FMSSafetyProviderConfig,
)
from llama_stack.providers.remote.safety.fms.detectors.base import (
    BaseDetector,
    DetectorProvider,
)
from llama_stack.providers.remote.safety.fms.detectors.chat import (
    ChatDetector,
)
from llama_stack.providers.remote.safety.fms.detectors.content import (
    ContentDetector,
)

# Type aliases for better readability
ConfigType = Union[ContentDetectorConfig, ChatDetectorConfig, FMSSafetyProviderConfig]
DetectorType = Union[BaseDetector, DetectorProvider]


class DetectorConfigError(ValueError):
    """Raised when detector configuration is invalid"""

    pass


async def get_adapter_impl(
    config: ConfigType,
    _deps: Dict[str, Any] = None,
) -> DetectorType:
    """Get appropriate detector implementation(s) based on config type.

    Args:
        config: Detector configuration object
        _deps: Optional dependencies for testing/injection

    Returns:
        Configured detector implementation

    Raises:
        DetectorConfigError: If configuration is invalid
        ValueError: If config type is not supported
    """
    try:
        # Handle provider config with multiple detectors
        if isinstance(config, FMSSafetyProviderConfig):
            detectors: Dict[str, DetectorType] = {}

            for detector_id, detector_config in config.detectors.items():
                if not isinstance(
                    detector_config, (ChatDetectorConfig, ContentDetectorConfig)
                ):
                    raise DetectorConfigError(
                        f"Invalid detector config type for {detector_id}: {type(detector_config)}"
                    )

                impl = await get_adapter_impl(detector_config, _deps)
                detectors[detector_id] = impl

            return DetectorProvider(detectors)

        # Handle single detector config
        if isinstance(config, ChatDetectorConfig):
            impl = ChatDetector(config)
        elif isinstance(config, ContentDetectorConfig):
            impl = ContentDetector(config)
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        await impl.initialize()
        return impl

    except Exception as e:
        raise DetectorConfigError(
            f"Failed to create detector implementation: {str(e)}"
        ) from e


__all__ = [
    # Factory
    "get_adapter_impl",
    # Configurations
    "ContentDetectorConfig",
    "ChatDetectorConfig",
    "FMSSafetyProviderConfig",
    "EndpointType",
    "DetectorParams",
    # Implementations
    "ChatDetector",
    "ContentDetector",
    "BaseDetector",
    "DetectorProvider",
    # Types
    "ConfigType",
    "DetectorType",
    "DetectorConfigError",
]
