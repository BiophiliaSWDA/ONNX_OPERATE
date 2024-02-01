import os

ROOT_PATH = '/Users/wuduo/Documents/BioWork/DifferentialTestingofDLFrameworks/ONNX_OPERATE/onnx2x'

DATASET_PATH = os.path.join(ROOT_PATH, 'database')

OP_TYPES_SIMILARITY_PATH = os.path.join(DATASET_PATH, 'op_types_similarity.json')

OPS_INFOS_PATH = os.path.join(DATASET_PATH, 'ops_infos.json')

OPS_MAPPINGS_PATH = os.path.join(DATASET_PATH, 'op_types_mapping.json')

PARAMETER_INFOS_PATH = os.path.join(DATASET_PATH, 'parameter_infos.json')

PARAMETER_MAPPINGS_PATH = os.path.join(DATASET_PATH, 'parameter_mapping.json')

TEST_CODE_PATH = os.path.join(ROOT_PATH, 'test_code')
