# -*- coding: utf-8 -*-

"""
@Title   : 获取ms和paddle的参数infos填入parameter_infos.json
@Time    : 2024/2/24 19:26
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""
import json
import re

import requests
from bs4 import BeautifulSoup

from onnx2x.config.config_path import OPS_MAPPINGS_PATH, PARAMETER_INFOS_PATH

ms_root_url = "https://www.mindspore.cn/docs/en/r2.2/api_python"
pd_root_url = "https://www.paddlepaddle.org.cn/documentation/docs/en/api"

def get_params_infos_ms(api):
    dtype = api.split('.')[1]
    ms_url = f"{ms_root_url}/{dtype}/{api}.html"
    print(ms_url)
    soup = get_response(ms_url)

    dts = soup.select('dl > dt')
    function = None
    for dt in dts:
        function = dt.text.replace('\n', ' ').strip()[:-9]
        break

    dds = soup.select('dl > dd > dl > dd')
    param_dict = {}
    for dd in dds:
        lis = dd.select('li')
        for li in lis:
            parameter_name = li.find_all('strong')
            if parameter_name:
                parameter_name = parameter_name[0].text
            else:
                continue
            descp = li.text.replace('\n', ' ')
            dtype = re.search(r'\((.*?)\)', descp)
            isRequired = False

            if dtype:
                dtype = dtype.group(1)

            if function:
                isRequired = False if f'{parameter_name}=' in function else True

            descp = descp.split('–')[-1].strip()
            param_dict[parameter_name] = set_pa_dict(descp, dtype, isRequired)

        break

    return param_dict


def get_params_infos_pd(api):
    api_ = api.replace('.', '/')

    pd_url = f"{pd_root_url}/{api_}_en.html"
    print(pd_url)
    soup = get_response(pd_url)

    dts = soup.select('dl > dt')
    function = None
    for dt in dts:
        function = dt.text.replace('\n', ' ').strip()[:-9]
        break

    dts = soup.select('dl > dd > dl > dt')
    param_dict = {}
    for dt in dts:
        if dt and dt.text.strip() == "Parameters":
            dd = dt.find_next()
            ps = dd.select('p')
            for p in ps:
                parameter_name = p.find_all('strong')
                if parameter_name:
                    parameter_name = parameter_name[0].text
                else:
                    continue

                descp = p.text.replace('\n', ' ')
                dtype = re.search(r'\((.*?)\)', descp)
                isRequired = False

                if dtype:
                    dtype = dtype.group(1)

                if function:
                    isRequired = False if f'{parameter_name}=' in function else True

                descp = descp.split('–')[-1].strip()
                param_dict[parameter_name] = set_pa_dict(descp, dtype, isRequired)

            break

    return param_dict


def set_pa_dict(descp=None, type=None, isRequired=None):
    return dict([('def', descp),
                 ('type', type),
                 ('isRequired', isRequired)])


def get_response(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    response.close()
    return soup


if __name__ == "__main__":
    with open(OPS_MAPPINGS_PATH, 'r') as f:
        ops_mappings = json.load(f)
    ms_params_infos = {}
    for onnx2x in ops_mappings.items():
        onnx_name = onnx2x[0]
        ms_api = onnx2x[1]["mindspore"]["apis"]
        ms_dict = {}
        for api in ms_api:
            param_dict = get_params_infos_ms(api)
            ms_dict[api] = param_dict
        ms_params_infos[onnx_name] = ms_dict
    # print(ms_params_infos)

    pd_params_infos = {}
    for onnx2x in ops_mappings.items():
        onnx_name = onnx2x[0]
        pd_api = onnx2x[1]["paddlepaddle"]["apis"]
        pd_dict = {}
        for api in pd_api:
            param_dict = get_params_infos_pd(api)
            pd_dict[api] = param_dict
        pd_params_infos[onnx_name] = pd_dict
    # print(pd_params_infos)

    with open(PARAMETER_INFOS_PATH, 'r') as f:
        parameter_infos = json.load(f)

    for onnx2x in parameter_infos.items():
        onnx_name = onnx2x[0]
        onnx2x[1]["paddlepaddle"] = pd_params_infos[onnx_name]
        onnx2x[1]["mindspore"] = ms_params_infos[onnx_name]

    with open(PARAMETER_INFOS_PATH, 'w') as f:
        json.dump(parameter_infos, f, sort_keys=True)