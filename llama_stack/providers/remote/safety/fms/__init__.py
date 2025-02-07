from typing import Any, Union
from .config import ContentDetectionConfig, ChatDetectionConfig


async def get_adapter_impl(
    config: Union[ContentDetectionConfig, ChatDetectionConfig], _deps
) -> Any:
    if isinstance(config, ChatDetectionConfig):
        from .chat_detector import ChatDetection

        impl = ChatDetection(config)
    else:
        # Handle ContentDetectionConfig with detectors list
        from .content_detector import ContentDetection

        impl = ContentDetection(config)

    await impl.initialize()
    return impl


__all__ = ["get_adapter_impl", "ContentDetectionConfig", "ChatDetectionConfig"]
