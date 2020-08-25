# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory

import onnx
import onnx.utils

from . import repository
from ..models.data_format import as_model_config_data_format
from ..models.data_type import DataType
from ..models.sources.onnx_model_file import ONNXModelFile
from ..models.targets.openvino_model import OpenvinoModel
from ..openvino_util import Config, execute_optimize_action
from ..protos.generated.model_config_pb2 import ModelInput, ModelOutput


def _onnx_data_type_to_tf_data_type(onnx_data_type):
    return DataType.from_onnx_data_type(onnx_data_type).to_tf_data_type()


def _get_tensor_value_info(graph, name: str):
    sub1, sub2 = name.split(':')
    is_input_port = True
    if sub1.isdigit() and not sub2.isdigit():
        node_name, port = sub2, sub1
    else:
        node_name, port = sub1, sub2 or '0'
        is_input_port = False

    get_info_name_func = _get_input_name_from_node if is_input_port is True else _get_output_name_from_node
    info_name = get_info_name_func(graph, node_name, int(port))
    value_info_list = list(graph.value_info) + list(graph.input) + list(graph.output)
    value_info = next(info for info in value_info_list if info.name == info_name)
    data_type = _onnx_data_type_to_tf_data_type(value_info.type.tensor_type.elem_type)
    shape = value_info.type.tensor_type.shape
    return [i.dim_value for i in shape.dim], data_type


def _get_input_name_from_node(graph, node_name: str, port: int):
    try:
        node = next(i for i in graph.node if i.name == node_name)
    except StopIteration as exception:
        raise ValueError('Unable to find the node.') from exception
    node_input_name = node.input[port]
    return node_input_name


def _get_output_name_from_node(graph, node_name: str, port: int):
    try:
        node = next(i for i in graph.node if i.name == node_name)
    except StopIteration as exception:
        raise ValueError('Unable to find the node.') from exception
    node_input_name = node.output[port]
    return node_input_name


def _get_inputs(graph, config):
    if config.input_info is None:
        return None
    inputs = []
    for name, data_format in config.input_info:
        shape, data_type = _get_tensor_value_info(graph, name)
        dims = [-1 if dim is None else dim for dim in shape[1:]]
        inputs.append(ModelInput(name=name,
                                 data_type=data_type,
                                 format=as_model_config_data_format(data_format),
                                 dims=dims))
    return inputs


def _get_outputs(graph, config):
    if config.output_names is None:
        return None
    outputs = []
    for name in config.output_names:
        shape, data_type = _get_tensor_value_info(graph, name+':0')
        outputs.append(ModelOutput(name=name,
                                   data_type=data_type,
                                   dims=[-1 if dim is None else dim for dim in shape[1:]]))
    return outputs


def _get_optimize_params(input_model, output_dir, max_batch_size, inputs, outputs):
    params = {'script_name': 'mo_onnx.py',
              'model_name': 'model',
              'input_model': input_model,
              'output_dir': output_dir}
    if inputs is not None:
        params['input'] = ','.join([i.name for i in inputs])
        params['input_shape'] = ','.join(str([max_batch_size] + list(i.dims)) for i in inputs)
    if outputs is not None:
        params['output'] = ','.join([i.name for i in outputs])
    return params


@repository.REPOSITORY.register(source_type=ONNXModelFile, target_type=OpenvinoModel, config_type=Config)
def compile_source(source: ONNXModelFile, config: Config) -> OpenvinoModel:
    model_proto = onnx.utils.polish_model(onnx.load(source.model_path))
    inputs = _get_inputs(model_proto.graph, config)  # pylint: disable=no-member
    outputs = _get_outputs(model_proto.graph, config)  # pylint: disable=no-member
    temp_path = TemporaryDirectory()
    optimize_params = _get_optimize_params(source.model_path, temp_path.name,
                                           config.max_batch_size, inputs, outputs)
    execute_optimize_action(optimize_params)
    return OpenvinoModel(inputs, outputs, temp_path)