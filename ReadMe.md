## Version2.0

### 预计更新
#### 1 `op_types_mapping.json`

    1.1 ✅ `onnx2paddlepaddle`的API信息
    1.2 ✅ `onnx2mindspore`的API信息

#### 2 `op_types_similarity.json`

    2.1 ✅ 更新选出的57个算子的相似度

#### 3 `parameter_infos.json`

基于`1`中的信息更新以下内容：

    3.1 `onnx2paddlepaddle`的参数信息
    3.2 `onnx2mindspore`的参数信息

#### 4 `parameter_mapping.json`

    4.1 `onnx2paddlepaddle`的参数匹配的结果
    4.2 `onnx2mindspore`的参数匹配的结果

#### 5 其他问题

- [ ] torch.nn.BatchNorm2d tf.keras.layers.BatchNormalization输出的结果不相同
- [ ] 卷积直接翻译的结果不一致
- [ ] Dropout函数的输出结果不一致
- [ ] Elu 的 alpha值在ms框架只能是1，其他框架可以为其他值，但是实际传入为1
- [ ] 算子Gemm是否为dense函数
- [ ] mindspore.nn.InstanceNorm2d仅支持在GPU上运行。可能也需要单独实现
- [ ] tf.keras.layers.LayerNormalization与其他框架中的函数参数不同
- [ ] LogSoftmax/Softmax算子在实现的时候需要制定dim(axis)，不然torch的结果与其他两个框架表现不一致性
- [ ] MaxPool
    - [ ] paddlepaddle中没有dilation参数
    - [ ] 可能需要单独实现pad等参数
- [ ] MaxUnpool存疑
- [ ] Reshape
