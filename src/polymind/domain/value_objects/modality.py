"""Modality value object — identifies input type."""

from enum import Enum


class Modality(str, Enum):
    """The type of input modality for a query."""

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    TABLE = "table"
    MULTI = "multi"
