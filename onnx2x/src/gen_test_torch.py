# -*- coding: utf-8 -*-

"""
@Title   : 
@Time    : 2024/2/1
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""
import os.path
import re
import subprocess
import time

from onnx2x.src.Database import Database

from onnx2x.config.config_path import TEST_CODE_PATH


d = Database()
d.set_framework('pytorch')
op_list = d.get_onnx_list()

tensor_pattern = [
    'api(tensor, *args)',
    'api(tensor+, *args)',
    'api([tensor+], *args)',
    'api(*args)(tensor)'
]


head = """import torch

tensor_pattern = [
    'api(tensor, *args)', 
    'api(tensor+, *args)', 
    'api([tensor+], *args)', 
    'api(*args)(tensor)'
]

torch_tensor1 = torch.randn(1, 4)
torch_tensor2 = torch.randn(1, 1, 4)
torch_tensor3 = torch.randn(1, 1, 1, 4)
torch_tensor4 = torch.randn(1, 1, 1, 1, 4)
"""

tensors = [
    'torch_tensor1',
    'torch_tensor2',
    'torch_tensor3',
    'torch_tensor4'
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
    elif pattern == tensor_pattern[3]:
        # 'api(*args)(tensor)'
        code = f'{start}' \
               f'   {api_name} = {api}({parameters})({tensor})\n' \
               f'{end}\n'
    elif pattern == tensor_pattern[2]:
        # 'api([tensor+], *args)'
        code = f'{start}' \
               f'   {api_name} = {api}([{tensor}, {tensor}], {parameters})\n' \
               f'{end}\n'
    else:
        code = ''

    return code


def gen_code():
    body = ''
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
            num_pattern = re.compile(r'\dd', re.S)
            dim = num_pattern.findall(api)
            if dim:
                dim = dim[0].replace('d', '')
                tensor = tensors[int(dim)]
            else:
                tensor = tensors[2]
            i = apis.index(api)
            pattern = patterns[i]

            requireds = d.get_onnx2x_para_requireds(onnx, api)

            code = gen_test(api, pattern, requireds, tensor, i)

            body += code

    with open(os.path.join(TEST_CODE_PATH, 'torch_test_code.py'), 'w') as torch_test_code:
        torch_test_code.write(head)
        torch_test_code.write('\n')
        torch_test_code.write(body)
        torch_test_code.write('\n')


# gen_code()


def gen_test_(api, pattern, requireds, tensor, i) -> str:
    api_name = api.replace('.', '_') + '_' + i.__str__() + '_' + tensor
    parameters = ', '.join(['1' for _ in range(requireds)])

    if pattern == tensor_pattern[0]:
        # 'api(tensor, *args)'
        code = f'{api_name} = {api}({tensor}, {parameters})\n'
    elif pattern == tensor_pattern[1]:
        # 'api(tensor+, *args)
        code = f'{api_name} = {api}({tensor}, {tensor}, {parameters})\n'
    elif pattern == tensor_pattern[3]:
        # 'api(*args)(tensor)'
        code = f'{api_name} = {api}({parameters})({tensor})\n'
    elif pattern == tensor_pattern[2]:
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
            num_pattern = re.compile(r'\dd', re.S)
            dim = num_pattern.findall(api)
            if dim:
                dim = dim[0].replace('d', '')
                tensor = tensors[int(dim)]
            else:
                tensor = tensors[2]
            i = apis.index(api)
            pattern = patterns[i]

            requireds = d.get_onnx2x_para_requireds(onnx, api)

            code = gen_test_(api, pattern, requireds, tensor, i)

            api_test_code_path = f'{TEST_CODE_PATH}/torch/{api}.py'
            with open(api_test_code_path, 'w') as f:
                f.write(head)
                f.write('\n')
                f.write(code)

            print('写入至文件', api_test_code_path)

            try:
                run_result = subprocess.run(['python', api_test_code_path])
                full_table_mapping = True if run_result.returncode == 0 else None

                # 强制更新full_table_mapping
                d.update_full_table_mapping(onnx, full_table_mapping, force=True)
            except:
                pass

# update_full_table_mapping_()
