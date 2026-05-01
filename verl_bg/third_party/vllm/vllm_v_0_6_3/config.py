# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/config.py

import copy
import enum
import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

from transformers import PretrainedConfig

# Add for verl
import vllm.config as _vllm_config
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.loader import BaseModelLoader

logger = init_logger(__name__)

# Patch vllm's _get_and_verify_max_len to handle models (e.g. Qwen3) whose
# rope_scaling dict does not contain the legacy "factor" key expected by vllm 0.6.3.
_orig_get_and_verify_max_len = _vllm_config._get_and_verify_max_len


def _patched_get_and_verify_max_len(hf_config, max_model_len, *args, **kwargs):
    rope_scaling = getattr(hf_config, 'rope_scaling', None)
    if isinstance(rope_scaling, dict) and 'factor' not in rope_scaling:
        hf_config = copy.copy(hf_config)
        hf_config.rope_scaling = {**rope_scaling, 'factor': 1.0}
    return _orig_get_and_verify_max_len(hf_config, max_model_len, *args, **kwargs)


_vllm_config._get_and_verify_max_len = _patched_get_and_verify_max_len

# Register Qwen3ForCausalLM in vllm 0.6.3's model registry.
# vllm 0.6.3 may copy _MODELS into a singleton instance (ModelRegistry) at init
# time, so we must patch both the module-level dicts AND all instance-level dicts
# on the ModelRegistry singleton. We iterate every dict attribute that contains
# Qwen2ForCausalLM and add Qwen3ForCausalLM with the same value.
def _register_qwen3_in_vllm_registry():
    try:
        from vllm.model_executor.models import registry as _reg_mod
        from vllm.model_executor.models.registry import ModelRegistry as _mr_inst

        _objs = [_reg_mod, _mr_inst, type(_mr_inst)]
        for _obj in _objs:
            try:
                _obj_dict = vars(_obj)
            except TypeError:
                continue
            for _name, _val in list(_obj_dict.items()):
                if (isinstance(_val, dict)
                        and 'Qwen2ForCausalLM' in _val
                        and 'Qwen3ForCausalLM' not in _val):
                    _val['Qwen3ForCausalLM'] = _val['Qwen2ForCausalLM']
                    logger.info("Registered Qwen3ForCausalLM in vllm registry attr '%s' of %s", _name, _obj)
    except Exception as _e:
        logger.warning("Could not register Qwen3ForCausalLM in vllm registry: %s", _e)


_register_qwen3_in_vllm_registry()


# Patch vllm's get_rope to handle Qwen3's rope_scaling type="default".
# vllm 0.6.3 does not recognise "default" as a valid scaling type; it is
# semantically equivalent to no scaling (plain RoPE), so we strip it out.
# We patch both the rotary_embedding module AND every model module that may
# have already imported get_rope into its own namespace (e.g. qwen2.py).
def _patch_get_rope_for_qwen3():
    try:
        from vllm.model_executor.layers import rotary_embedding as _rot_mod
        _orig_get_rope = _rot_mod.get_rope

        def _patched_get_rope(*args, **kwargs):
            rope_scaling = kwargs.get('rope_scaling')
            if isinstance(rope_scaling, dict) and rope_scaling.get('type') == 'default':
                kwargs['rope_scaling'] = None
            return _orig_get_rope(*args, **kwargs)

        _rot_mod.get_rope = _patched_get_rope

        # Also fix any vllm model module already in sys.modules that bound
        # get_rope via `from ... import get_rope`.
        for _mod_name, _mod in list(sys.modules.items()):
            if (_mod_name.startswith('vllm.model_executor.models')
                    and getattr(_mod, 'get_rope', None) is _orig_get_rope):
                setattr(_mod, 'get_rope', _patched_get_rope)
    except Exception as _e:
        logger.warning("Could not patch get_rope for Qwen3: %s", _e)


_patch_get_rope_for_qwen3()


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = "dtensor"
    DUMMY_HF = "dummy_hf"
    DUMMY_MEGATRON = "dummy_megatron"
    DUMMY_DTENSOR = "dummy_dtensor"


class ModelConfig(ModelConfig):

    def __init__(self, hf_config: PretrainedConfig, *args, **kwargs) -> None:
        super().__init__(model=hf_config._name_or_path, tokenizer=hf_config._name_or_path, *args, **kwargs)
        self.hf_config = hf_config


@dataclass
class LoadConfig:
    """
    download_dir: Directory to download and load the weights, default to the
        default cache directory of huggingface.
    load_format: The format of the model weights to load:
        "auto" will try to load the weights in the safetensors format and
            fall back to the pytorch bin format if safetensors format is
            not available.
        "pt" will load the weights in the pytorch bin format.
        "safetensors" will load the weights in the safetensors format.
        "npcache" will load the weights in pytorch format and store
            a numpy cache to speed up the loading.
        "dummy" will initialize the weights with random values, which is
            mainly for profiling.
        "tensorizer" will use CoreWeave's tensorizer library for
            fast weight loading.
        "bitsandbytes" will load nf4 type weights.
    ignore_patterns: The list of patterns to ignore when loading the model.
        Default to "original/**/*" to avoid repeated loading of llama's
        checkpoints.

    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info("Ignoring the following patterns when downloading weights: %s", self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in LoadFormat.__members__ if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(f"load format '{load_format}' is not supported in ROCm. "
                             f"Supported load formats are "
                             f"{rocm_supported_load_format}")
