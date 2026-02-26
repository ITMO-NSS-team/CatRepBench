from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from genbench.representations.utils.category_encoders_wrapper import (
    CategoryEncodersRepresentationBase,
)


@dataclass
class SumRepresentation(CategoryEncodersRepresentationBase):
    name: str = "sum_representation"
    ENCODER_CLS_NAME: ClassVar[str] = "SumEncoder"
