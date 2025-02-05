# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any, Union
from .config import FMSModelConfig, FMSChatAdapterConfig


async def get_adapter_impl(
    config: Union[FMSModelConfig, FMSChatAdapterConfig], _deps
) -> Any:
    if isinstance(config, FMSChatAdapterConfig):
        from .chat_detector import FMSChatAdapter

        impl = FMSChatAdapter(config)
    elif config.detector_id == "mmluTopicMatch":
        from .topic_detector import FMSTopicMatchAdapter

        impl = FMSTopicMatchAdapter(config)
    else:
        from .content_detector import FMSModelAdapter

        impl = FMSModelAdapter(config)

    await impl.initialize()
    return impl


__all__ = ["get_adapter_impl", "FMSModelConfig", "FMSChatConfig"]
