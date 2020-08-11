# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory

import tensorflow as tf

from . import repository
from ..models.data_format import as_model_config_data_format, DataFormat
from ..models.sources.tf_frozen_graph_file import FrozenGraphFile
from ..models.targets.openvino_model import OpenvinoModel
from ..openvino_util import Config, execute_optimize_action
from ..protos.generated.model_config_pb2 import ModelInput, ModelOutput
from ..utilities import get_tensor_by_fuzzy_name


def _get_inputs(graph, config):
    if config.input_info is None:
        return None
    inputs = []
    for name, data_format in config.input_info:
        tensor = get_tensor_by_fuzzy_name(graph, name)
        # OpenVINO only support NCHW, so should transpose shape if data_format is 'channels_last'
        dims = [-1 if dim is None else dim for dim in tensor.shape[1:]]
        if data_format == DataFormat.CHANNELS_LAST:
            data_format = DataFormat.CHANNELS_FIRST
            channel = dims.pop(-1)
            dims.insert(0, channel)
        inputs.append(ModelInput(name=name,
                                 data_type=tensor.dtype.as_datatype_enum,
                                 format=as_model_config_data_format(data_format),
                                 dims=dims))
    return inputs


def _get_outputs(graph, config):
    if config.output_names is None:
        return None
    outputs = []
    for name in config.output_names:
        tensor = get_tensor_by_fuzzy_name(graph, name)
        outputs.append(ModelOutput(name=name,
                                   data_type=tensor.dtype.as_datatype_enum,
                                   dims=[-1 if dim is None else dim for dim in tensor.shape[1:]]))
    return outputs


def _get_optimize_params(input_model, output_dir, max_batch_size, inputs, outputs):
    params = {'script_name': 'mo_tf.py',
              'model_name': 'model',
              'input_model': input_model,
              'output_dir': output_dir,
              'batch': str(max_batch_size)}
    if inputs is not None:
        params['input'] = ','.join(i.name for i in inputs)
    if outputs is not None:
        params['output'] = ','.join(i.name for i in outputs)
    return params


@repository.REPOSITORY.register(source_type=FrozenGraphFile, target_type=OpenvinoModel, config_type=Config)
def compile_source(source: FrozenGraphFile, config: Config) -> OpenvinoModel:
    graph_def = tf.compat.v1.GraphDef()
    with open(source.model_path, 'rb') as graph_file:
        graph_def.ParseFromString(graph_file.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        inputs = _get_inputs(graph, config)
        outputs = _get_outputs(graph, config)

    temp_path = TemporaryDirectory()
    optimize_params = _get_optimize_params(source.model_path, temp_path.name,
                                           config.max_batch_size, inputs, outputs)
    execute_optimize_action(optimize_params)
    return OpenvinoModel(inputs, outputs, temp_path)
