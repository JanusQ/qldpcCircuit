"""
QLDPC Enumeration Decoder with CSC optimization support

This package provides efficient QLDPC decoding using enumeration methods
with optional CSC (Compressed Sparse Column) matrix optimization for sparse matrices.
"""

from .enumlikelydecoder import EnumDecoder, EnumDecoderCSC

__version__ = "1.0.0"
__author__ = "QLDPC Team"

__all__ = ["EnumDecoder", "EnumDecoderCSC"]
