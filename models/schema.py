
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    model_path: str
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, list[Any]]
    input_dtypes: dict[str, str]
    output_shapes: dict[str, list[Any]]
    output_dtypes: dict[str, str]
    available_providers: list[str] = field(default_factory=list)
    compatible_providers: list[str] = field(default_factory=list)
    observed_execution_providers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
