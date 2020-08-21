# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import TestCase

import tensorflow as tf

import model_compiler.compilers.saved_model_to_openvino_model as compiler
from model_compiler.compilers.saved_model_to_openvino_model import Config
from model_compiler.models.targets.saved_model import DataFormat, Input, ModelInput, Output, SavedModel


def _make_saved_model() -> SavedModel:
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as session:
        input_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 3, 4], name='x')
        input_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 3, 4], name='y')
        weight = tf.Variable(initial_value=4.2, dtype=tf.float32)
        output_z = tf.multiply(input_x + input_y, weight, name='z')

        session.run(weight.initializer)

    return SavedModel(inputs=[Input(name='x', tensor=input_x, data_format=DataFormat.CHANNELS_FIRST),
                              Input(name='y', tensor=input_y, data_format=DataFormat.CHANNELS_FIRST)],
                      outputs=[Output(name='z', tensor=output_z)],
                      session=session)


class CompileSourceTestCase(TestCase):
    def test_compile_simple(self):
        config = Config.from_json({'input_names': ['x', 'y'],
                                   'output_names': ['z'],
                                   'input_formats': ['channels_first', 'channels_first'],
                                   'max_batch_size': 1})
        compiled = compiler.compile_source(source=_make_saved_model(), config=config)
        self.assertEqual([model_input.name for model_input in compiled.inputs], ['x', 'y'])
        self.assertEqual([model_input.format for model_input in compiled.inputs],
                         [ModelInput.FORMAT_NCHW, ModelInput.FORMAT_NCHW])  # pylint: disable=no-member
        self.assertEqual([model_output.name for model_output in compiled.outputs], ['z'])
