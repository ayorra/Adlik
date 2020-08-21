# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory
from . import repository
from ..models.targets.saved_model import SavedModel
from ..models.targets.openvino_model import OpenvinoModel
from ..openvino_util import Config, execute_optimize_action


def _get_optimize_params(input_model_dir, output_dir, max_batch_size, inputs, outputs):
    params = {'script_name': 'mo_tf.py',
              'model_name': 'model',
              'saved_model_dir': input_model_dir,
              'output_dir': output_dir,
              'batch': str(max_batch_size),
              'input': ','.join(i.name for i in inputs),
              'output': ','.join(i.name for i in outputs)}
    return params


@repository.REPOSITORY.register(source_type=SavedModel, target_type=OpenvinoModel, config_type=Config)
def compile_source(source: SavedModel, config: Config) -> OpenvinoModel:
    with TemporaryDirectory() as directory:
        source.save(directory)
        openvino_temp_path = TemporaryDirectory()
        inputs = source.get_inputs()
        outputs = source.get_outputs()
        optimize_params = _get_optimize_params(directory, openvino_temp_path.name,
                                               config.max_batch_size, inputs, outputs)
        execute_optimize_action(optimize_params)
    return OpenvinoModel(inputs, outputs, openvino_temp_path)
