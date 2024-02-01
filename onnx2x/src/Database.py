# -*- coding: utf-8 -*-

"""
@Title   : 
@Time    : 2024/1/29 15:29
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""
import json

from onnx2x.config import config_path as p
from onnx2x.config import op_types as op


class Database:
    def __init__(self):
        self.__framework = 'pytorch'
        self.__onnx2x_api_mappings = {}
        self.__onnx2x_para_mappings = {}
        self.__onnx2x_para_infos = {}
        self.__onnx_list = op.LAYER_OP_TYPES

        # self.__update_full_table_mapping(init=False)
        self.__refresh_database()

    def set_framework(self, framework):
        self.__framework = framework

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

    def check(self):
        print('INFO: CHECK API MAPPINGS...\nFRAMEWORK: {}'.format(self.__framework))
        self.__check_api_mappings()
        print('INFO: CHECK PARAMETER INFOMATION...\nFRAMEWORK: {}'.format(self.__framework))
        self.__check_para_infos()
        print('INFO: CHECK PARAMETER MAPPINGS...\nFRAMEWORK: {}'.format(self.__framework))
        self.__check_para_mappings()

    def __check_api_mappings(self):
        _onnx2x_api_mappings = self.__onnx2x_api_mappings
        _onnx2x_para_mappings = self.__onnx2x_para_mappings
        _onnx2x_para_infos = self.__onnx2x_para_infos

        for _onnx in _onnx2x_api_mappings.items():
            _op_type = _onnx[0]
            skip = ['GRU', 'LSTM', 'RNN']
            if _op_type in skip:
                continue
            _apis = _onnx[1][self.__framework]['apis']
            for _api in _apis:
                if _api not in _onnx2x_para_mappings[_op_type][self.__framework].keys():
                    print(f'参数映射表中没有{_op_type}算子在{self.__framework}框架下的{_api}')

                if _api not in _onnx2x_para_infos[_op_type][self.__framework].keys():
                    print(f'参数信息表中没有{_op_type}算子在{self.__framework}框架下的{_api}')

    def __check_para_infos(self):
        _onnx2x_api_mappings = self.__onnx2x_api_mappings
        _onnx2x_para_mappings = self.__onnx2x_para_mappings
        _onnx2x_para_infos = self.__onnx2x_para_infos

        for _onnx in _onnx2x_para_infos.items():
            _op_type = _onnx[0]
            skip = ['GRU', 'LSTM', 'RNN']
            if _op_type in skip:
                continue
            _apis = _onnx[1][self.__framework].keys()
            for _api in _apis:
                if _api not in _onnx2x_para_mappings[_op_type][self.__framework].keys():
                    print(f'参数映射表中没有{_op_type}算子在{self.__framework}框架下的{_api}')

                if _api not in _onnx2x_api_mappings[_op_type][self.__framework]['apis']:
                    print(f'函数映射表中没有{_op_type}算子在{self.__framework}框架下的{_api}')

    def __check_para_mappings(self):
        _onnx2x_api_mappings = self.__onnx2x_api_mappings
        _onnx2x_para_mappings = self.__onnx2x_para_mappings
        _onnx2x_para_infos = self.__onnx2x_para_infos

        for _onnx in _onnx2x_para_mappings.items():
            _op_type = _onnx[0]
            skip = ['GRU', 'LSTM', 'RNN']
            if _op_type in skip:
                continue
            _apis = _onnx[1][self.__framework].keys()
            for _api in _apis:
                if _api not in _onnx2x_para_infos[_op_type][self.__framework].keys():
                    print(f'参数信息表中没有{_op_type}算子在{self.__framework}框架下的{_api}')

                if _api not in _onnx2x_api_mappings[_op_type][self.__framework]['apis']:
                    print(f'函数映射表中没有{_op_type}算子在{self.__framework}框架下的{_api}')

    def get_onnx_list(self):
        return self.__onnx_list

    def get_onnx2x_apis(self, onnx: str):
        return self.__onnx2x_api_mappings[onnx][self.__framework]['apis']

    def get_onnx2x_patterns(self, onnx: str):
        return self.__onnx2x_api_mappings[onnx][self.__framework]['patterns']

    def get_all_patterns(self) -> list:
        _patterns = set()
        for _op_type in self.__onnx_list:
            _tmp_patterns = self.get_onnx2x_patterns(_op_type)
            for _pattern in _tmp_patterns:
                _patterns.add(_pattern)

        return list(_patterns)

    def get_onnx2x_para_requireds(self, onnx: str, api: str) -> int:
        requireds = 0
        tmp_dict = self.__onnx2x_para_infos[onnx][self.__framework][api]
        for _ in tmp_dict.items():
            if 'isRequired' in _[1].keys():
                if _[1]['isRequired'] and _[1]['type'] not in ['tensor', 'tensors']:
                    requireds += 1

        return requireds

    def get_full_table_mapping(self, onnx):
        return self.__onnx2x_api_mappings[onnx][self.__framework]['full_table_mapping']

    def update(self, init: bool):
        self.__update_full_table_mapping(init)

    def __update_full_table_mapping(self, init):
        if init:
            for _onnx in self.__onnx2x_api_mappings:
                _op_type = _onnx[0]
                self.__onnx2x_api_mappings[_onnx][self.__framework]['full_table_mapping'] = None

            with open(p.OPS_MAPPINGS_PATH, 'w') as onnx2x_api_mappings_file:
                json.dump(self.__onnx2x_api_mappings, onnx2x_api_mappings_file, sort_keys=True)

            self.__read_onnx2x_api_mappings()

        self.__update_apis_full_table_mapping()
        self.__update_pad_full_table_mapping()

    def __update_pad_full_table_mapping(self):
        for _onnx in self.__onnx2x_para_mappings.items():
            _op_type = _onnx[0]
            _para_dict = _onnx[1][self.__framework]
            for _api in _para_dict.items():
                _api_name = _api[0]
                _onnx_para_list = _api[1].keys()
                if 'pads' in _onnx_para_list or 'auto_pad' in _onnx_para_list:
                    self.update_full_table_mapping(_op_type, False)

    def __update_apis_full_table_mapping(self):
        for _onnx in self.__onnx2x_para_infos.items():
            _op_type = _onnx[0]
            _full_table_mapping = None if list(_onnx[1][self.__framework]) == [] else True
            self.update_full_table_mapping(_op_type, full_table_mapping=_full_table_mapping)

    def update_full_table_mapping(self, onnx, full_table_mapping, force=False):
        _org_full_table_mapping = self.__onnx2x_api_mappings[onnx][self.__framework]['full_table_mapping']
        _new_full_table_mapping = full_table_mapping
        _flag = False
        if not force:
            if _org_full_table_mapping is None:
                _flag = True
            elif _org_full_table_mapping:
                if not _new_full_table_mapping:
                    _flag = True
            elif not _org_full_table_mapping:
                pass

        if _flag or force:
                self.__onnx2x_api_mappings[onnx][self.__framework]['full_table_mapping'] = _new_full_table_mapping

        with open(p.OPS_MAPPINGS_PATH, 'w') as onnx2x_api_mappings_file:
            json.dump(self.__onnx2x_api_mappings, onnx2x_api_mappings_file, sort_keys=True)

        self.__read_onnx2x_api_mappings()


d = Database()

# d.check()
# d.set_framework('tensorflow') # default: pytorch

# d.update(True)
# print(d.get_onnx2x_apis('Add'))
# print(d.get_onnx2x_patterns('Add'))
# print(d.get_onnx2x_para_requireds('AveragePool', 'tf.keras.layers.AveragePooling1D'))
# print(d.get_all_patterns())