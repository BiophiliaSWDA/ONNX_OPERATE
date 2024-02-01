# -*- coding: utf-8 -*-

"""
@Title   : 
@Time    : 2024/1/29 16:07
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""
import re
import subprocess
import time

from onnx2x.src.Database import Database

from onnx2x.config.config_path import TEST_CODE_PATH

d = Database()
op_list = d.get_onnx_list()

tensor_pattern = [
    'api(tensor, *args)',
    'api(tensor+, *args)',
    'api(*args)(tensor)',
    'api(*args)([tensor+])',
    'api([tensor+], *args)',
]


head = """import tensorflow as tf

tensor_pattern = [
    'api(tensor, *args)'
    'api(tensor+, *args)',
    'api(*args)(tensor)',
    'api(*args)([tensor+])',
    'api([tensor+], *args)',
]

tf_tensor1 = tf.ones(shape=(1, 4), dtype=tf.float32)
tf_tensor2 = tf.ones(shape=(1, 1, 4), dtype=tf.float32)
tf_tensor3 = tf.ones(shape=(1, 1, 1, 4), dtype=tf.float32)
tf_tensor4 = tf.ones(shape=(1, 1, 1, 1, 4), dtype=tf.float32)
"""

tensors = [
    'tf_tensor1',
    'tf_tensor2',
    'tf_tensor3',
    'tf_tensor4'
]


def gen_test(api, pattern, requireds, tensor, i) -> str:
    api_name = api.replace('.', '_') + '_' + i.__str__() + '_' + tensor
    parameters = ', '.join(['1' for _ in range(requireds)])

    start = f'# {api} {pattern}\n' \
            f'num = {time.time().__str__()}\n' \
            f'try:\n'
    end = f'   print({api_name})\n' \
          f'except Exception as e: \n' \
          f'   print(num, e)\n'
    if pattern == tensor_pattern[0]:
        # 'api(tensor, *args)'
        code = f'{start}' \
               f'   {api_name} = {api}({tensor}, {parameters})\n' \
               f'{end}\n'
    elif pattern == tensor_pattern[1]:
        # 'api(tensor+, *args)
        code = f'{start}' \
               f'   {api_name} = {api}({tensor}, {tensor}, {parameters})\n' \
               f'{end}\n'
    elif pattern == tensor_pattern[2]:
        # 'api(*args)(tensor)'
        code = f'{start}' \
               f'   {api_name} = {api}({parameters})({tensor})\n' \
               f'{end}\n'
    elif pattern == tensor_pattern[3]:
        # 'api(*args)([tensor+])'
        code = f'{start}' \
               f'   {api_name} = {api}({parameters})([{tensor}, {tensor}])\n' \
               f'{end}\n'
    elif pattern == tensor_pattern[4]:
        # 'api([tensor+], *args)'
        code = f'{start}' \
               f'   {api_name} = {api}([{tensor}, {tensor}], {parameters})\n' \
               f'{end}\n'
    else:
        code = ''

    return code


def gen_test_(api, pattern, requireds, tensor, i) -> str:
    api_name = api.replace('.', '_') + '_' + i.__str__() + '_' + tensor
    parameters = ', '.join(['1' for _ in range(requireds)])

    if pattern == tensor_pattern[0]:
        # 'api(tensor, *args)'
        code = f'{api_name} = {api}({tensor}, {parameters})\n'
    elif pattern == tensor_pattern[1]:
        # 'api(tensor+, *args)
        code = f'{api_name} = {api}({tensor}, {tensor}, {parameters})\n'
    elif pattern == tensor_pattern[2]:
        # 'api(*args)(tensor)'
        code = f'{api_name} = {api}({parameters})({tensor})\n'
    elif pattern == tensor_pattern[3]:
        # 'api(*args)([tensor+])'
        code = f'{api_name} = {api}({parameters})([{tensor}, {tensor}])\n'
    elif pattern == tensor_pattern[4]:
        # 'api([tensor+], *args)'
        code = f'{api_name} = {api}([{tensor}, {tensor}], {parameters})\n'
    else:
        code = ''

    return code


def update_full_table_mapping_():
    skip = ['GRU', 'LSTM', 'RNN']
    for onnx in op_list:
        if onnx in skip:
            continue
        apis = d.get_onnx2x_apis(onnx)
        patterns = d.get_onnx2x_patterns(onnx)
        # for p in patterns:
        #     tmp_pattern.add(p)
        # continue

        for api in apis:
            num_pattern = re.compile(r'\dD', re.S)
            dim = num_pattern.findall(api)
            if dim:
                dim = dim[0].replace('D', '')
                tensor = tensors[int(dim)]
            else:
                tensor = tensors[2]
            i = apis.index(api)
            pattern = patterns[i]

            requireds = d.get_onnx2x_para_requireds(onnx, api)

            code = gen_test_(api, pattern, requireds, tensor, i)

            api_test_code_path = f'{TEST_CODE_PATH}/{api}.py'
            with open(api_test_code_path, 'w') as f:
                f.write(head)
                f.write('\n')
                f.write(code)

            print('写入至文件', api_test_code_path)

            try:
                subprocess.run(['python', api_test_code_path])
                full_table_mapping = True
            except:
                full_table_mapping = None

            d.update_full_table_mapping(onnx, full_table_mapping)


# update_full_table_mapping_()
