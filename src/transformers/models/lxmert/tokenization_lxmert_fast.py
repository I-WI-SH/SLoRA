# coding=utf-8
# Copyright 2020 The Google AI Team, Stanford University and The HuggingFace Inc. team.
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

from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_lxmert import LxmertTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "unc-nlp/lxmert-base-uncased": "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "unc-nlp/lxmert-base-uncased": (
            "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/tokenizer.json"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "unc-nlp/lxmert-base-uncased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "unc-nlp/lxmert-base-uncased": {"do_lower_case": True},
}


class LxmertTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" LXMERT tokenizer (backed by HuggingFace's *tokenizers* library).

    [`LxmertTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = LxmertTokenizer
