# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import open_clip
from torch import Tensor, nn


class ClipTokenizer(nn.Module):
    """Wrap an OpenCLIP tokenizer for a MobileCLIP configuration."""

    def __init__(self, cfg, *args, **kwargs):
        """Initialize the configured OpenCLIP tokenizer."""
        super().__init__()
        self.context_length = cfg["text_cfg"]["context_length"]
        model_name = getattr(cfg["text_cfg"], "open_clip_tokenizer", "ViT-B-16")
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return len(self.tokenizer.encoder)

    def get_encodings(self) -> dict[str, int]:
        """Return the tokenizer encoding map."""
        return self.tokenizer.encoder

    def get_eot_token(self) -> int:
        """Return the end-of-text token ID."""
        # Tokenizing an empty string returns a list [sot_id, eot_id]
        return self.tokenizer("")[1]

    def get_sot_token(self) -> int:
        """Return the start-of-text token ID."""
        # Tokenizing an empty string returns a list [sot_id, eot_id]
        return self.tokenizer("")[0]

    def forward(self, input_sentence: str, *args, **kwargs) -> Tensor:
        """Tokenize a sentence to the configured context length."""
        # tokenizer returns indices as a string
        tokenized_sentence = self.tokenizer(input_sentence, self.context_length)
        assert tokenized_sentence.shape[-1] == self.context_length, (
            "Tokenized tensor should be exactly `context_length` long."
        )
        return tokenized_sentence
