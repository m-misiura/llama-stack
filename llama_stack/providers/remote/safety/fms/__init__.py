from typing import Any, Union
from .config import FMSModelConfig, FMSChatAdapterConfig


async def get_adapter_impl(
    config: Union[FMSModelConfig, FMSChatAdapterConfig], _deps
) -> Any:
    if isinstance(config, FMSChatAdapterConfig):
        from .chat_detector import FMSChatAdapter

        impl = FMSChatAdapter(config)
    else:
        # Handle FMSModelConfig with detectors list
        from .content_detector import FMSModelAdapter

        impl = FMSModelAdapter(config)

    await impl.initialize()
    return impl


__all__ = ["get_adapter_impl", "FMSModelConfig", "FMSChatConfig"]
