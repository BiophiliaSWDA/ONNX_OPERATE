# -*- coding: utf-8 -*-

"""
@Title   : 
@Time    : 2024/1/29 15:29
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""
import json

from config import config_path as p
from config import op_types as op


class Database:
    def __init__(self):
        self.__framework = 'tensorflow'
        self.__onnx2x_api_mappings = {}
        self.__onnx2x_para_mappings = {}
        self.__onnx2x_para_infos = {}
        self.__onnx_list = op.LAYER_OP_TYPES

        self.__refresh_database()

    def __refresh_database(self):
        self.__read_onnx2x_api_info()
        self.__read_onnx2x_api_mappings()
        self.__read_onnx2x_para_infos()
        self.__read_onnx2x_para_mappings()

    def __read_onnx2x_api_mappings(self):
        with open(p.OPS_MAPPINGS_PATH, 'r') as onnx2x_api_mappings_file:
            self.__onnx2x_api_mappings = json.load(onnx2x_api_mappings_file)

    def __read_onnx2x_api_info(self):
        pass

    def __read_onnx2x_para_mappings(self):
        with open(p.PARAMETER_MAPPINGS_PATH, 'r') as onnx2x_para_mappings_file:
            self.__onnx2x_para_mappings = json.load(onnx2x_para_mappings_file)

    def __read_onnx2x_para_infos(self):
        with open(p.PARAMETER_INFOS_PATH, 'r') as onnx2x_para_infos_file:
            self.__onnx2x_para_infos = json.load(onnx2x_para_infos_file)

    def get_onnx_list(self):
        return self.__onnx_list

    def get_onnx2x_apis(self, onnx: str):
        return self.__onnx2x_api_mappings[onnx][self.__framework]['apis']

    def get_onnx2x_patterns(self, onnx: str):
        return self.__onnx2x_api_mappings[onnx][self.__framework]['patterns']

    def get_onnx2x_para_requireds(self, onnx: str, api: str) -> int:
        requireds = 0
        tmp_dict = self.__onnx2x_para_infos[onnx][self.__framework][api]
        for _ in tmp_dict.items():
            if 'isRequired' in _[1].keys():
                if _[1]['isRequired']:
                    requireds += 1

        return requireds


# print(d.get_onnx2x_apis('Add'))
# print(d.get_onnx2x_patterns('Add'))
# print(d.get_onnx2x_para_requireds('AveragePool', 'tf.keras.layers.AveragePooling1D'))
