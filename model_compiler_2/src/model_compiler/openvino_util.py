# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess  # nosec
import sys
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from .models.data_format import DataFormat, str_to_data_format
from .utilities import split_by


def _get_input_info(input_names, str_formats):
    if input_names is None:
        return None
    if str_formats is None:
        return [(input_name, None) for input_name in input_names]
    if len(input_names) == len(str_formats):
        return list(zip(input_names, map(str_to_data_format, str_formats)))
    raise ValueError('Input names and formats should be have the same length or have no formats')


class Config(NamedTuple):
    input_info: Optional[Sequence[Tuple[str, Optional[DataFormat]]]]
    output_names: Optional[Sequence[str]]
    max_batch_size: int

    @staticmethod
    def from_json(value: Mapping[str, Any]) -> 'Config':
        # When compile onnx to openvino, need provide port:node_name for input_names
        # Example: '0:node_name1,1:node_name1'
        input_names = value.get('input_names')
        input_formats = value.get('input_formats')

        # When compile onnx to openvino, need provide node_name for output_names
        # Example: 'node_name1,node_name2'
        output_names = value.get('output_names')
        max_batch_size = value.get('max_batch_size', 1)
        input_info = _get_input_info(input_names, input_formats)
        return Config(input_info=input_info,
                      output_names=output_names,
                      max_batch_size=max_batch_size)

    @staticmethod
    def from_env(env: Mapping[str, Any]) -> 'Config':
        input_names = split_by(env.get('INPUT_NAMES'), ',')
        input_formats = split_by(env.get('INPUT_FORMATS'), ',')
        output_names = split_by(env.get('OUTPUT_NAMES'), ',')
        max_batch_size = env.get('MAX_BATCH_SIZE', '1')
        input_info = _get_input_info(input_names, input_formats)
        return Config(input_info=input_info,
                      output_names=output_names,
                      max_batch_size=int(max_batch_size))


def get_version():
    version_txt = os.path.join(_acquire_optimizer_base_dir(), 'deployment_tools/model_optimizer/version.txt')
    with open(version_txt) as file:
        version = file.readline().rstrip()
    return version


def execute_optimize_action(params: Dict[str, str]):
    subprocess.run(_args_dict_to_list(params), check=True)  # nosec


def _args_dict_to_list(params: Dict[str, str]) -> List[str]:
    args = [sys.executable, _acquire_optimizer_script_dir(params.pop('script_name'))]
    for key, value in params.items():
        args.extend(['--' + key] if value == '' else ['--' + key, value])
    return args


def _acquire_optimizer_script_dir(script_name):
    return os.path.join(_acquire_optimizer_base_dir(), 'deployment_tools/model_optimizer/{}'.format(script_name))


def _acquire_optimizer_base_dir():
    return os.getenv('INTEL_CVSDK_DIR', '/opt/intel/openvino_2020.4.287')
