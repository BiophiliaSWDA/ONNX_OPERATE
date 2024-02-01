## 文档说明

### `op_types_mapping.json`

- 状态：**可能需要更新**，以及mindspore和paddlepaddle数据的更新
- 内容：`apis`, `full_table_mapping`, `patterns`
- 用途：**函数映射**
- 可能涉及的函数：	
    1. ONNX2X API获取，`apis`	
    2. API的分类，`full_table_mapping`
    3. API模版获取，`patterns`

### `parameter_mapping.json`

- 状态：**可能需要更新**，以及mindspore和paddlepaddle数据的更新
- 内容：parameter
- 用途：**参数对齐**
- 可能涉及的函数：
    1. 参数获取

### `op_types_similarity.json`

- 状态：**待更新**，但可以通过一些设置来使用
- 内容：ONNX算子的相似度
- 用涂：**API变异**
- 可能涉及的函数：
    1. API Mutate

### `parameter_infos.json`

- 状态：**可能需要更新，以及mindspore和paddlepaddle数据的更新**
- 内容：ONNX参数信息，以及具体框架的参数信息（其中包含参数是否为必须参数`isRequired`）
- 用途：用于识别在框架下参数是否为必须参数，其他关键字均是数据处理部分需要的关键字
- 可能涉及的函数：
    1.  翻译必须参数（解决方案【有待商榷】：将必须参数设置为默认值或者单独实现【`full_table_mapping`==true】）

### `ops_infos.json`

- 状态：**待更新**，但可以通过一些设置来使用
- 内容：ONNX算子的基本信息
- 用途：用于数据处理
- 在变异的过程中**无**可能涉及的参数