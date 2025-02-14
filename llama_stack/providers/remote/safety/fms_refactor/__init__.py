from typing import Any, Union
from .config import ContentDetectorConfig, ChatDetectorConfig
from .chat_detector import ChatDetector
from .content_detector import ContentDetector


async def get_adapter_impl(
    config: Union[ContentDetectorConfig, ChatDetectorConfig], _deps: Any = None
) -> Any:
    """Get appropriate detector implementation based on config type"""
    if isinstance(config, ChatDetectorConfig):
        impl = ChatDetector(config)
    else:
        impl = ContentDetector(config)

    await impl.initialize()
    return impl


__all__ = [
    "get_adapter_impl",
    "ContentDetectorConfig",
    "ChatDetectorConfig",
    "ChatDetector",
    "ContentDetector",
]
