# Copyright 2019 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from tempfile import NamedTemporaryFile
from unittest import TestCase
import torch
import model_compiler.compilers.onnx_model_file_to_openvino_model as compiler
from model_compiler.compilers.onnx_model_file_to_openvino_model import Config, ModelInput
from model_compiler.models.data_format import DataFormat
from model_compiler.models.sources.onnx_model_file import ONNXModelFile


class ConfigTestCase(TestCase):
    def test_from_json(self):
        self.assertEqual(Config.from_json({'input_names': ['0:node_name'],
                                           'input_formats': ['channels_first'],
                                           'output_names': ['node_name'],
                                           'max_batch_size': 1}),
                         Config(input_info=[('0:node_name', DataFormat.CHANNELS_FIRST)],
                                output_names=['node_name'],
                                max_batch_size=1))

    def test_from_env(self):
        self.assertEqual(Config.from_env({'INPUT_NAMES': '0:node_name1,0:node_name2',
                                          'OUTPUT_NAMES': 'node_name',
                                          'INPUT_FORMATS': 'channels_first,channels_first',
                                          'MAX_BATCH_SIZE': '1'}),
                         Config(input_info=[('0:node_name1', DataFormat.CHANNELS_FIRST),
                                            ('0:node_name2', DataFormat.CHANNELS_FIRST)],
                                output_names=['node_name'],
                                max_batch_size=1))


class Mnist(torch.nn.Module):  # pylint: disable=abstract-method
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, input_x):
        temp = torch.nn.functional.relu(self.conv1(input_x))
        temp = torch.nn.functional.max_pool2d(temp, 2, 2)
        temp = torch.nn.functional.relu(self.conv2(temp))
        temp = torch.nn.functional.max_pool2d(temp, 2, 2)
        temp = temp.view(-1, 4 * 4 * 50)
        temp = torch.nn.functional.relu(self.fc1(temp))
        temp = self.fc2(temp)
        return torch.nn.functional.softmax(temp, dim=-1)


def _save_onnx_model(file_name):
    model = Mnist()
    dummy_input = torch.randn(1, 1, 28, 28)  # pylint: disable=no-member
    torch.onnx.export(model, dummy_input, file_name, verbose=True, keep_initializers_as_inputs=True)


class CompileSourceTestCase(TestCase):
    def test_compile_with_variables(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'input_names': ['0:Conv_0', '0:Relu_9'],
                                       'output_names': ['Softmax_11'],
                                       'input_formats': ['channels_first', 'channels_first'],
                                       'max_batch_size': 1})
            compiled = compiler.compile_source(ONNXModelFile(model_path=model_file.name), config)

        self.assertEqual([model_input.name for model_input in compiled.inputs], ['0:Conv_0', '0:Relu_9'])
        self.assertEqual([model_input.format for model_input in compiled.inputs],
                         [ModelInput.FORMAT_NCHW, ModelInput.FORMAT_NCHW])  # pylint: disable=no-member
        self.assertEqual([model_output.name for model_output in compiled.outputs], ['Softmax_11'])

    def test_compile_with_variables_channel_first(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'input_names': ['0:Conv_0', '0:Relu_9'],
                                       'output_names': ['Softmax_11'],
                                       'input_formats': ['channels_first', 'channels_first'],
                                       'max_batch_size': 1})
            compiled = compiler.compile_source(ONNXModelFile(model_path=model_file.name), config)

        self.assertEqual([model_input.name for model_input in compiled.inputs], ['0:Conv_0', '0:Relu_9'])
        self.assertEqual([model_input.format for model_input in compiled.inputs],
                         [ModelInput.FORMAT_NCHW, ModelInput.FORMAT_NCHW])  # pylint: disable=no-member
        self.assertEqual([model_output.name for model_output in compiled.outputs], ['Softmax_11'])

    def test_compile_with_variables_channel_last(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'input_names': ['0:Conv_0', '0:Relu_9'],
                                       'output_names': ['Softmax_11'],
                                       'input_formats': ['channels_last', 'channels_last'],
                                       'max_batch_size': 1})
            compiled = compiler.compile_source(ONNXModelFile(model_path=model_file.name), config)

        self.assertEqual([model_input.name for model_input in compiled.inputs], ['0:Conv_0', '0:Relu_9'])
        self.assertEqual([model_input.format for model_input in compiled.inputs],
                         [ModelInput.FORMAT_NHWC, ModelInput.FORMAT_NHWC])  # pylint: disable=no-member
        self.assertEqual([model_output.name for model_output in compiled.outputs], ['Softmax_11'])

    def test_compile_with_variables_input_name_format_different_length(self):
        self.assertRaises(ValueError, Config.from_json,
                          {'input_names': ['0:Conv_0', '1:Relu_9'],
                           'output_names': ['Softmax_11'],
                           'input_formats': ['channels_last'],
                           'max_batch_size': 1})

    def test_compile_with_no_input_name(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'output_names': ['Softmax_11'],
                                       'max_batch_size': 1})
            compiled = compiler.compile_source(ONNXModelFile(model_path=model_file.name), config)
        self.assertEqual(compiled.inputs, None)

    def test_compile_with_no_input_formats(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'input_names': ['0:Conv_0'],
                                       'output_names': ['Softmax_11'],
                                       'max_batch_size': 1})
            compiled = compiler.compile_source(ONNXModelFile(model_path=model_file.name), config)
        self.assertEqual([model_input.format for model_input in compiled.inputs],
                         [ModelInput.FORMAT_NONE])  # pylint: disable=no-member
        self.assertEqual([model_input.name for model_input in compiled.inputs], ['0:Conv_0'])

    def test_compile_with_no_outputs(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'input_names': ['0:Conv_0'],
                                       'max_batch_size': 1})
            compiled = compiler.compile_source(ONNXModelFile(model_path=model_file.name), config)
        self.assertEqual([model_input.format for model_input in compiled.inputs],
                         [ModelInput.FORMAT_NONE])  # pylint: disable=no-member
        self.assertEqual([model_input.name for model_input in compiled.inputs], ['0:Conv_0'])

    def test_compile_with_error_get_info(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'input_names': ['0:xxxxxxx'],
                                       'max_batch_size': 1})
            self.assertRaises(ValueError, compiler.compile_source,
                              ONNXModelFile(model_path=model_file.name),
                              config)

    def test_compile_with_error_get_input_name_from_node(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'input_names': ['0:xxx'],
                                       'max_batch_size': 1})
            self.assertRaises(ValueError, compiler.compile_source,
                              ONNXModelFile(model_path=model_file.name),
                              config)

    def test_compile_with_error_get_output_name_from_node(self):
        with NamedTemporaryFile(suffix='.onnx') as model_file:
            _save_onnx_model(model_file.name)
            config = Config.from_json({'input_names': ['0:Conv_0'],
                                       'output_names': ['xxx'],
                                       'max_batch_size': 1})
            self.assertRaises(ValueError, compiler.compile_source,
                              ONNXModelFile(model_path=model_file.name),
                              config)
