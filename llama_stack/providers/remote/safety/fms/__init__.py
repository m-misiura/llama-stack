# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from .config import FMSModelConfig


async def get_adapter_impl(config: FMSModelConfig, _deps) -> Any:
    from .content_detector import FMSModelAdapter

    impl = FMSModelAdapter(config)
    await impl.initialize()
    return impl
