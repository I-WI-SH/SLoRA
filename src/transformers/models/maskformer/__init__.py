# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available


_import_structure = {"configuration_maskformer": ["MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "MaskFormerConfig"]}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_maskformer"] = ["MaskFormerFeatureExtractor"]


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_maskformer"] = [
        "MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MaskFormerForInstanceSegmentation",
        "MaskFormerModel",
        "MaskFormerPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_maskformer import MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, MaskFormerConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_maskformer import MaskFormerFeatureExtractor
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_maskformer import (
            MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            MaskFormerForInstanceSegmentation,
            MaskFormerModel,
            MaskFormerPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
