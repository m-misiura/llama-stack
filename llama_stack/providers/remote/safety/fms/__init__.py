from typing import Any, Union
from .config import FMSModelConfig, ChatDetectionConfig


async def get_adapter_impl(
    config: Union[FMSModelConfig, ChatDetectionConfig], _deps
) -> Any:
    if isinstance(config, ChatDetectionConfig):
        from .chat_detector import ChatDetection

        impl = ChatDetection(config)
    else:
        # Handle FMSModelConfig with detectors list
        from .content_detector import FMSModelAdapter

        impl = FMSModelAdapter(config)

    await impl.initialize()
    return impl


__all__ = ["get_adapter_impl", "FMSModelConfig", "ChatDetectionConfig"]
