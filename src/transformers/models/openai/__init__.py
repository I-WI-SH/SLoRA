# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_openai": ["OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OpenAIGPTConfig"],
    "tokenization_openai": ["OpenAIGPTTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_openai_fast"] = ["OpenAIGPTTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_openai"] = [
        "OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OpenAIGPTDoubleHeadsModel",
        "OpenAIGPTForSequenceClassification",
        "OpenAIGPTLMHeadModel",
        "OpenAIGPTModel",
        "OpenAIGPTPreTrainedModel",
        "load_tf_weights_in_openai_gpt",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_openai"] = [
        "TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFOpenAIGPTDoubleHeadsModel",
        "TFOpenAIGPTForSequenceClassification",
        "TFOpenAIGPTLMHeadModel",
        "TFOpenAIGPTMainLayer",
        "TFOpenAIGPTModel",
        "TFOpenAIGPTPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
    from .tokenization_openai import OpenAIGPTTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_openai_fast import OpenAIGPTTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_openai import (
            OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OpenAIGPTDoubleHeadsModel,
            OpenAIGPTForSequenceClassification,
            OpenAIGPTLMHeadModel,
            OpenAIGPTModel,
            OpenAIGPTPreTrainedModel,
            load_tf_weights_in_openai_gpt,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_openai import (
            TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFOpenAIGPTDoubleHeadsModel,
            TFOpenAIGPTForSequenceClassification,
            TFOpenAIGPTLMHeadModel,
            TFOpenAIGPTMainLayer,
            TFOpenAIGPTModel,
            TFOpenAIGPTPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
