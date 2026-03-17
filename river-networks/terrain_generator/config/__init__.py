"""Configuration utilities and preset management."""

from .presets import PresetManager, PresetError
from .erosion_params import (
    ErosionParameterSet,
    RockLayerConfig,
    load_erosion_parameters,
    save_erosion_parameters,
    normalize_layer_inputs,
)

__all__ = [
    'PresetManager',
    'PresetError',
    'ErosionParameterSet',
    'RockLayerConfig',
    'load_erosion_parameters',
    'save_erosion_parameters',
    'normalize_layer_inputs',
]
