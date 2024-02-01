import torch

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

# torch.add api(tensor+, *args)
num = 1706792732.986803
try:
   torch_add_0_torch_tensor3 = torch.add(torch_tensor3, torch_tensor3, )
   print(torch_add_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.AvgPool1d api(*args)(tensor)
num = 1706792732.986819
try:
   torch_nn_AvgPool1d_0_torch_tensor2 = torch.nn.AvgPool1d(1)(torch_tensor2)
   print(torch_nn_AvgPool1d_0_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.nn.AvgPool2d api(*args)(tensor)
num = 1706792732.986824
try:
   torch_nn_AvgPool2d_1_torch_tensor3 = torch.nn.AvgPool2d(1)(torch_tensor3)
   print(torch_nn_AvgPool2d_1_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.AvgPool3d api(*args)(tensor)
num = 1706792732.986828
try:
   torch_nn_AvgPool3d_2_torch_tensor4 = torch.nn.AvgPool3d(1)(torch_tensor4)
   print(torch_nn_AvgPool3d_2_torch_tensor4)
except Exception as e: 
   print(num, e)

# torch.nn.BatchNorm1d api(*args)(tensor)
num = 1706792732.986832
try:
   torch_nn_BatchNorm1d_0_torch_tensor2 = torch.nn.BatchNorm1d(1)(torch_tensor2)
   print(torch_nn_BatchNorm1d_0_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.nn.BatchNorm2d api(*args)(tensor)
num = 1706792732.986835
try:
   torch_nn_BatchNorm2d_1_torch_tensor3 = torch.nn.BatchNorm2d(1)(torch_tensor3)
   print(torch_nn_BatchNorm2d_1_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.BatchNorm3d api(*args)(tensor)
num = 1706792732.986838
try:
   torch_nn_BatchNorm3d_2_torch_tensor4 = torch.nn.BatchNorm3d(1)(torch_tensor4)
   print(torch_nn_BatchNorm3d_2_torch_tensor4)
except Exception as e: 
   print(num, e)

# torch.ceil api(tensor, *args)
num = 1706792732.9868412
try:
   torch_ceil_0_torch_tensor3 = torch.ceil(torch_tensor3, )
   print(torch_ceil_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.CELU api(*args)(tensor)
num = 1706792732.9868438
try:
   torch_nn_CELU_0_torch_tensor3 = torch.nn.CELU()(torch_tensor3)
   print(torch_nn_CELU_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.cat api([tensor+], *args)
num = 1706792732.986847
try:
   torch_cat_0_torch_tensor3 = torch.cat([torch_tensor3, torch_tensor3], )
   print(torch_cat_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Conv3d api(*args)(tensor)
num = 1706792732.9868522
try:
   torch_nn_Conv3d_0_torch_tensor4 = torch.nn.Conv3d(1, 1, 1)(torch_tensor4)
   print(torch_nn_Conv3d_0_torch_tensor4)
except Exception as e: 
   print(num, e)

# torch.nn.Conv1d api(*args)(tensor)
num = 1706792732.986856
try:
   torch_nn_Conv1d_1_torch_tensor2 = torch.nn.Conv1d(1, 1, 1)(torch_tensor2)
   print(torch_nn_Conv1d_1_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.nn.Conv2d api(*args)(tensor)
num = 1706792732.986859
try:
   torch_nn_Conv2d_2_torch_tensor3 = torch.nn.Conv2d(1, 1, 1)(torch_tensor3)
   print(torch_nn_Conv2d_2_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.ConvTranspose1d api(*args)(tensor)
num = 1706792732.9868631
try:
   torch_nn_ConvTranspose1d_0_torch_tensor2 = torch.nn.ConvTranspose1d(1, 1, 1)(torch_tensor2)
   print(torch_nn_ConvTranspose1d_0_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.nn.ConvTranspose2d api(*args)(tensor)
num = 1706792732.986866
try:
   torch_nn_ConvTranspose2d_1_torch_tensor3 = torch.nn.ConvTranspose2d(1, 1, 1)(torch_tensor3)
   print(torch_nn_ConvTranspose2d_1_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.ConvTranspose3d api(*args)(tensor)
num = 1706792732.98687
try:
   torch_nn_ConvTranspose3d_2_torch_tensor4 = torch.nn.ConvTranspose3d(1, 1, 1)(torch_tensor4)
   print(torch_nn_ConvTranspose3d_2_torch_tensor4)
except Exception as e: 
   print(num, e)

# torch.nn.functional.dropout api(tensor, *args)
num = 1706792732.9868731
try:
   torch_nn_functional_dropout_0_torch_tensor3 = torch.nn.functional.dropout(torch_tensor3, )
   print(torch_nn_functional_dropout_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Dropout api(*args)(tensor)
num = 1706792732.9868748
try:
   torch_nn_Dropout_1_torch_tensor3 = torch.nn.Dropout()(torch_tensor3)
   print(torch_nn_Dropout_1_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Dropout2d api(*args)(tensor)
num = 1706792732.986878
try:
   torch_nn_Dropout2d_2_torch_tensor3 = torch.nn.Dropout2d()(torch_tensor3)
   print(torch_nn_Dropout2d_2_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Dropout1d api(*args)(tensor)
num = 1706792732.98688
try:
   torch_nn_Dropout1d_3_torch_tensor2 = torch.nn.Dropout1d()(torch_tensor2)
   print(torch_nn_Dropout1d_3_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.nn.Dropout3d api(*args)(tensor)
num = 1706792732.9868832
try:
   torch_nn_Dropout3d_4_torch_tensor4 = torch.nn.Dropout3d()(torch_tensor4)
   print(torch_nn_Dropout3d_4_torch_tensor4)
except Exception as e: 
   print(num, e)

# torch.nn.ELU api(*args)(tensor)
num = 1706792732.9868858
try:
   torch_nn_ELU_0_torch_tensor3 = torch.nn.ELU()(torch_tensor3)
   print(torch_nn_ELU_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.erf api(tensor, *args)
num = 1706792732.986888
try:
   torch_erf_0_torch_tensor3 = torch.erf(torch_tensor3, )
   print(torch_erf_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Flatten api(*args)(tensor)
num = 1706792732.986891
try:
   torch_nn_Flatten_0_torch_tensor3 = torch.nn.Flatten()(torch_tensor3)
   print(torch_nn_Flatten_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.floor api(tensor, *args)
num = 1706792732.9868941
try:
   torch_floor_0_torch_tensor3 = torch.floor(torch_tensor3, )
   print(torch_floor_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.GELU api(*args)(tensor)
num = 1706792732.9868958
try:
   torch_nn_GELU_0_torch_tensor3 = torch.nn.GELU()(torch_tensor3)
   print(torch_nn_GELU_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Linear api(*args)(tensor)
num = 1706792732.986899
try:
   torch_nn_Linear_0_torch_tensor3 = torch.nn.Linear(4, 1)(torch_tensor3)
   print(torch_nn_Linear_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Hardsigmoid api(*args)(tensor)
num = 1706792732.986903
try:
   torch_nn_Hardsigmoid_0_torch_tensor3 = torch.nn.Hardsigmoid()(torch_tensor3)
   print(torch_nn_Hardsigmoid_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Hardswish api(*args)(tensor)
num = 1706792732.986905
try:
   torch_nn_Hardswish_0_torch_tensor3 = torch.nn.Hardswish()(torch_tensor3)
   print(torch_nn_Hardswish_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.InstanceNorm2d api(*args)(tensor)
num = 1706792732.986908
try:
   torch_nn_InstanceNorm2d_0_torch_tensor3 = torch.nn.InstanceNorm2d(1)(torch_tensor3)
   print(torch_nn_InstanceNorm2d_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.InstanceNorm1d api(*args)(tensor)
num = 1706792732.986911
try:
   torch_nn_InstanceNorm1d_1_torch_tensor2 = torch.nn.InstanceNorm1d(1)(torch_tensor2)
   print(torch_nn_InstanceNorm1d_1_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.nn.InstanceNorm3d api(*args)(tensor)
num = 1706792732.9869142
try:
   torch_nn_InstanceNorm3d_2_torch_tensor4 = torch.nn.InstanceNorm3d(1)(torch_tensor4)
   print(torch_nn_InstanceNorm3d_2_torch_tensor4)
except Exception as e: 
   print(num, e)

# torch.nn.LocalResponseNorm api(*args)(tensor)
num = 1706792732.986917
try:
   torch_nn_LocalResponseNorm_0_torch_tensor3 = torch.nn.LocalResponseNorm(1)(torch_tensor3)
   print(torch_nn_LocalResponseNorm_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.LayerNorm api(*args)(tensor)
num = 1706792732.9869199
try:
   torch_nn_LayerNorm_0_torch_tensor3 = torch.nn.LayerNorm([1, 1, 1, 4])(torch_tensor3)
   print(torch_nn_LayerNorm_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.LeakyReLU api(*args)(tensor)
num = 1706792732.986923
try:
   torch_nn_LeakyReLU_0_torch_tensor3 = torch.nn.LeakyReLU()(torch_tensor3)
   print(torch_nn_LeakyReLU_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.LogSoftmax api(*args)(tensor)
num = 1706792732.986926
try:
   torch_nn_LogSoftmax_0_torch_tensor3 = torch.nn.LogSoftmax()(torch_tensor3)
   print(torch_nn_LogSoftmax_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.LPPool1d api(*args)(tensor)
num = 1706792732.986929
try:
   torch_nn_LPPool1d_0_torch_tensor2 = torch.nn.LPPool1d(2, 1)(torch_tensor2)
   print(torch_nn_LPPool1d_0_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.nn.LPPool2d api(*args)(tensor)
num = 1706792732.9869308
try:
   torch_nn_LPPool2d_1_torch_tensor3 = torch.nn.LPPool2d(2, 1)(torch_tensor3)
   print(torch_nn_LPPool2d_1_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.maximum api(tensor+, *args)
num = 1706792732.986934
try:
   torch_maximum_0_torch_tensor3 = torch.maximum(torch_tensor3, torch_tensor3, )
   print(torch_maximum_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.MaxPool2d api(*args)(tensor)
num = 1706792732.986937
try:
   torch_nn_MaxPool2d_0_torch_tensor3 = torch.nn.MaxPool2d(1)(torch_tensor3)
   print(torch_nn_MaxPool2d_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.MaxPool1d api(*args)(tensor)
num = 1706792732.98694
try:
   torch_nn_MaxPool1d_1_torch_tensor2 = torch.nn.MaxPool1d(1)(torch_tensor2)
   print(torch_nn_MaxPool1d_1_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.nn.MaxPool3d api(*args)(tensor)
num = 1706792732.986943
try:
   torch_nn_MaxPool3d_2_torch_tensor4 = torch.nn.MaxPool3d(1)(torch_tensor4)
   print(torch_nn_MaxPool3d_2_torch_tensor4)
except Exception as e: 
   print(num, e)

# torch.nn.MaxUnpool3d api(*args)(tensor)
num = 1706792732.986946
try:
   torch_nn_MaxUnpool3d_0_torch_tensor4 = torch.nn.MaxUnpool3d(1)(torch_tensor4)
   print(torch_nn_MaxUnpool3d_0_torch_tensor4)
except Exception as e: 
   print(num, e)

# torch.nn.MaxUnpool2d api(*args)(tensor)
num = 1706792732.9869492
try:
   torch_nn_MaxUnpool2d_1_torch_tensor3 = torch.nn.MaxUnpool2d(1)(torch_tensor3)
   print(torch_nn_MaxUnpool2d_1_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.MaxUnpool1d api(*args)(tensor)
num = 1706792732.9869509
try:
   torch_nn_MaxUnpool1d_2_torch_tensor2 = torch.nn.MaxUnpool1d(1)(torch_tensor2)
   print(torch_nn_MaxUnpool1d_2_torch_tensor2)
except Exception as e: 
   print(num, e)

# torch.minimum api(tensor+, *args)
num = 1706792732.986954
try:
   torch_minimum_0_torch_tensor3 = torch.minimum(torch_tensor3, torch_tensor3, )
   print(torch_minimum_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Mish api(*args)(tensor)
num = 1706792732.986957
try:
   torch_nn_Mish_0_torch_tensor3 = torch.nn.Mish()(torch_tensor3)
   print(torch_nn_Mish_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.fmod api(tensor+, *args)
num = 1706792732.98696
try:
   torch_fmod_0_torch_tensor3 = torch.fmod(torch_tensor3, torch_tensor3)
   print(torch_fmod_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.mul api(tensor+, *args)
num = 1706792732.986963
try:
   torch_mul_0_torch_tensor3 = torch.mul(torch_tensor3, torch_tensor3, )
   print(torch_mul_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.PReLU api(*args)(tensor)
num = 1706792732.9869661
try:
   torch_nn_PReLU_0_torch_tensor3 = torch.nn.PReLU()(torch_tensor3)
   print(torch_nn_PReLU_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.pow api(tensor+, *args)
num = 1706792732.986969
try:
   torch_pow_0_torch_tensor3 = torch.pow(torch_tensor3, torch_tensor3, )
   print(torch_pow_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.reciprocal api(tensor, *args)
num = 1706792732.986971
try:
   torch_reciprocal_0_torch_tensor3 = torch.reciprocal(torch_tensor3, )
   print(torch_reciprocal_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.max api(tensor, *args)
num = 1706792732.986974
try:
   torch_max_0_torch_tensor3 = torch.max(torch_tensor3, )
   print(torch_max_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.mean api(tensor, *args)
num = 1706792732.986977
try:
   torch_mean_0_torch_tensor3 = torch.mean(torch_tensor3, )
   print(torch_mean_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.min api(tensor, *args)
num = 1706792732.9869802
try:
   torch_min_0_torch_tensor3 = torch.min(torch_tensor3, )
   print(torch_min_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.prod api(tensor, *args)
num = 1706792732.9869819
try:
   torch_prod_0_torch_tensor3 = torch.prod(torch_tensor3, )
   print(torch_prod_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.sum api(tensor, *args)
num = 1706792732.986985
try:
   torch_sum_0_torch_tensor3 = torch.sum(torch_tensor3, )
   print(torch_sum_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.functional.relu api(tensor, *args)
num = 1706792732.986988
try:
   torch_nn_functional_relu_0_torch_tensor3 = torch.nn.functional.relu(torch_tensor3, )
   print(torch_nn_functional_relu_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.ReLU api(*args)(tensor)
num = 1706792732.98699
try:
   torch_nn_ReLU_1_torch_tensor3 = torch.nn.ReLU()(torch_tensor3)
   print(torch_nn_ReLU_1_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.reshape api(tensor, *args)
num = 1706792732.9869928
try:
   torch_reshape_0_torch_tensor3 = torch.reshape(torch_tensor3, [1, 1, 2, 2])
   print(torch_reshape_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.round api(tensor, *args)
num = 1706792732.986996
try:
   torch_round_0_torch_tensor3 = torch.round(torch_tensor3, )
   print(torch_round_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.SELU api(*args)(tensor)
num = 1706792732.986998
try:
   torch_nn_SELU_0_torch_tensor3 = torch.nn.SELU()(torch_tensor3)
   print(torch_nn_SELU_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Sigmoid api(*args)(tensor)
num = 1706792732.9870012
try:
   torch_nn_Sigmoid_0_torch_tensor3 = torch.nn.Sigmoid()(torch_tensor3)
   print(torch_nn_Sigmoid_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Softmax2d api(*args)(tensor)
num = 1706792732.9870038
try:
   torch_nn_Softmax2d_0_torch_tensor3 = torch.nn.Softmax2d()(torch_tensor3)
   print(torch_nn_Softmax2d_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Softmax api(*args)(tensor)
num = 1706792732.987006
try:
   torch_nn_Softmax_1_torch_tensor3 = torch.nn.Softmax()(torch_tensor3)
   print(torch_nn_Softmax_1_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Softplus api(*args)(tensor)
num = 1706792732.987008
try:
   torch_nn_Softplus_0_torch_tensor3 = torch.nn.Softplus()(torch_tensor3)
   print(torch_nn_Softplus_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Softsign api(*args)(tensor)
num = 1706792732.987011
try:
   torch_nn_Softsign_0_torch_tensor3 = torch.nn.Softsign()(torch_tensor3)
   print(torch_nn_Softsign_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.sub api(tensor+, *args)
num = 1706792732.987013
try:
   torch_sub_0_torch_tensor3 = torch.sub(torch_tensor3, torch_tensor3, )
   print(torch_sub_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Tanh api(*args)(tensor)
num = 1706792732.987016
try:
   torch_nn_Tanh_0_torch_tensor3 = torch.nn.Tanh()(torch_tensor3)
   print(torch_nn_Tanh_0_torch_tensor3)
except Exception as e: 
   print(num, e)

# torch.nn.Upsample api(*args)(tensor)
num = 1706792732.987019
try:
   torch_nn_Upsample_0_torch_tensor3 = torch.nn.Upsample(1)(torch_tensor3)
   print(torch_nn_Upsample_0_torch_tensor3)
except Exception as e: 
   print(num, e)


