
import tensorflow as tf

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

# tf.keras.layers.Add api(*args)([tensor+])
num = 1706759198.384992
try:
   tf_keras_layers_Add_0_tf_tensor2 = tf.keras.layers.Add()([tf_tensor2, tf_tensor2])
   print(tf_keras_layers_Add_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling2D api(*args)(tensor)
num = 1706759198.385014
try:
   tf_keras_layers_AveragePooling2D_0_tf_tensor2 = tf.keras.layers.AveragePooling2D()(tf_tensor2)
   print(tf_keras_layers_AveragePooling2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling3D api(*args)(tensor)
num = 1706759198.38502
try:
   tf_keras_layers_AveragePooling3D_1_tf_tensor3 = tf.keras.layers.AveragePooling3D()(tf_tensor3)
   print(tf_keras_layers_AveragePooling3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling1D api(*args)(tensor)
num = 1706759198.385024
try:
   tf_keras_layers_AveragePooling1D_2_tf_tensor1 = tf.keras.layers.AveragePooling1D()(tf_tensor1)
   print(tf_keras_layers_AveragePooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.BatchNormalization api(*args)(tensor)
num = 1706759198.385029
try:
   tf_keras_layers_BatchNormalization_0_tf_tensor2 = tf.keras.layers.BatchNormalization()(tf_tensor2)
   print(tf_keras_layers_BatchNormalization_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.ceil api(tensor, *args)
num = 1706759198.385033
try:
   tf_math_ceil_0_tf_tensor2 = tf.math.ceil(tf_tensor2, 1)
   print(tf_math_ceil_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.concat api([tensor+], *args)
num = 1706759198.385037
try:
   tf_concat_0_tf_tensor2 = tf.concat([tf_tensor2, tf_tensor2], 1, 1)
   print(tf_concat_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Concatenate api(*args)([tensor+])
num = 1706759198.3850398
try:
   tf_keras_layers_Concatenate_1_tf_tensor2 = tf.keras.layers.Concatenate()([tf_tensor2, tf_tensor2])
   print(tf_keras_layers_Concatenate_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2D api(*args)(tensor)
num = 1706759198.385045
try:
   tf_keras_layers_Conv2D_0_tf_tensor2 = tf.keras.layers.Conv2D(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3D api(*args)(tensor)
num = 1706759198.385049
try:
   tf_keras_layers_Conv3D_1_tf_tensor3 = tf.keras.layers.Conv3D(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1D api(*args)(tensor)
num = 1706759198.385053
try:
   tf_keras_layers_Conv1D_2_tf_tensor1 = tf.keras.layers.Conv1D(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1DTranspose api(*args)(tensor)
num = 1706759198.3850582
try:
   tf_keras_layers_Conv1DTranspose_0_tf_tensor1 = tf.keras.layers.Conv1DTranspose(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv1DTranspose_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3DTranspose api(*args)(tensor)
num = 1706759198.385063
try:
   tf_keras_layers_Conv3DTranspose_1_tf_tensor3 = tf.keras.layers.Conv3DTranspose(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv3DTranspose_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2DTranspose api(*args)(tensor)
num = 1706759198.385067
try:
   tf_keras_layers_Conv2DTranspose_2_tf_tensor2 = tf.keras.layers.Conv2DTranspose(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv2DTranspose_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Dropout api(*args)(tensor)
num = 1706759198.385071
try:
   tf_keras_layers_Dropout_0_tf_tensor2 = tf.keras.layers.Dropout(1)(tf_tensor2)
   print(tf_keras_layers_Dropout_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.elu api(tensor, *args)
num = 1706759198.385074
try:
   tf_keras_activations_elu_0_tf_tensor2 = tf.keras.activations.elu(tf_tensor2, 1)
   print(tf_keras_activations_elu_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ELU api(*args)(tensor)
num = 1706759198.385076
try:
   tf_keras_layers_ELU_1_tf_tensor2 = tf.keras.layers.ELU()(tf_tensor2)
   print(tf_keras_layers_ELU_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.nn.elu api(tensor, *args)
num = 1706759198.3850791
try:
   tf_nn_elu_2_tf_tensor2 = tf.nn.elu(tf_tensor2, 1)
   print(tf_nn_elu_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.erf api(tensor, *args)
num = 1706759198.385082
try:
   tf_math_erf_0_tf_tensor2 = tf.math.erf(tf_tensor2, 1)
   print(tf_math_erf_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Flatten api(*args)(tensor)
num = 1706759198.3850849
try:
   tf_keras_layers_Flatten_0_tf_tensor2 = tf.keras.layers.Flatten()(tf_tensor2)
   print(tf_keras_layers_Flatten_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.floor api(tensor, *args)
num = 1706759198.385088
try:
   tf_math_floor_0_tf_tensor2 = tf.math.floor(tf_tensor2, 1)
   print(tf_math_floor_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.gelu api(tensor, *args)
num = 1706759198.385091
try:
   tf_keras_activations_gelu_0_tf_tensor2 = tf.keras.activations.gelu(tf_tensor2, 1)
   print(tf_keras_activations_gelu_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.nn.gelu api(tensor, *args)
num = 1706759198.3850932
try:
   tf_nn_gelu_1_tf_tensor2 = tf.nn.gelu(tf_tensor2, 1)
   print(tf_nn_gelu_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling2D api(*args)(tensor)
num = 1706759198.385097
try:
   tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor2 = tf.keras.layers.GlobalAveragePooling2D()(tf_tensor2)
   print(tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling3D api(*args)(tensor)
num = 1706759198.3851001
try:
   tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor3 = tf.keras.layers.GlobalAveragePooling3D()(tf_tensor3)
   print(tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling1D api(*args)(tensor)
num = 1706759198.385102
try:
   tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor1 = tf.keras.layers.GlobalAveragePooling1D()(tf_tensor1)
   print(tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling3D api(*args)(tensor)
num = 1706759198.385105
try:
   tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor3 = tf.keras.layers.GlobalMaxPooling3D()(tf_tensor3)
   print(tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling1D api(*args)(tensor)
num = 1706759198.385108
try:
   tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor1 = tf.keras.layers.GlobalMaxPooling1D()(tf_tensor1)
   print(tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling2D api(*args)(tensor)
num = 1706759198.385111
try:
   tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor2 = tf.keras.layers.GlobalMaxPooling2D()(tf_tensor2)
   print(tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.raw_ops.LRN api(tensor, *args)
num = 1706759198.385115
try:
   tf_raw_ops_LRN_0_tf_tensor2 = tf.raw_ops.LRN(tf_tensor2, 1)
   print(tf_raw_ops_LRN_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LayerNormalization api(*args)(tensor)
num = 1706759198.385119
try:
   tf_keras_layers_LayerNormalization_0_tf_tensor2 = tf.keras.layers.LayerNormalization()(tf_tensor2)
   print(tf_keras_layers_LayerNormalization_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LeakyReLU api(*args)(tensor)
num = 1706759198.385122
try:
   tf_keras_layers_LeakyReLU_0_tf_tensor2 = tf.keras.layers.LeakyReLU()(tf_tensor2)
   print(tf_keras_layers_LeakyReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.maximum api(tensor+, *args)
num = 1706759198.3851252
try:
   tf_math_maximum_0_tf_tensor2 = tf.math.maximum(tf_tensor2, tf_tensor2, 1, 1)
   print(tf_math_maximum_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.maximum api(tensor+, *args)
num = 1706759198.385128
try:
   tf_keras_layers_maximum_1_tf_tensor2 = tf.keras.layers.maximum(tf_tensor2, tf_tensor2, 1)
   print(tf_keras_layers_maximum_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling3D api(*args)(tensor)
num = 1706759198.3851311
try:
   tf_keras_layers_MaxPooling3D_0_tf_tensor3 = tf.keras.layers.MaxPooling3D()(tf_tensor3)
   print(tf_keras_layers_MaxPooling3D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling2D api(*args)(tensor)
num = 1706759198.385134
try:
   tf_keras_layers_MaxPooling2D_1_tf_tensor2 = tf.keras.layers.MaxPooling2D()(tf_tensor2)
   print(tf_keras_layers_MaxPooling2D_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling1D api(*args)(tensor)
num = 1706759198.3851368
try:
   tf_keras_layers_MaxPooling1D_2_tf_tensor1 = tf.keras.layers.MaxPooling1D()(tf_tensor1)
   print(tf_keras_layers_MaxPooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.minimum api(tensor+, *args)
num = 1706759198.38514
try:
   tf_minimum_0_tf_tensor2 = tf.minimum(tf_tensor2, tf_tensor2, 1, 1)
   print(tf_minimum_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.mish api(tensor, *args)
num = 1706759198.385143
try:
   tf_keras_activations_mish_0_tf_tensor2 = tf.keras.activations.mish(tf_tensor2, 1)
   print(tf_keras_activations_mish_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.floormod api(tensor+, *args)
num = 1706759198.3851461
try:
   tf_math_floormod_0_tf_tensor2 = tf.math.floormod(tf_tensor2, tf_tensor2, 1, 1)
   print(tf_math_floormod_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.multiply api(tensor+, *args)
num = 1706759198.385149
try:
   tf_math_multiply_0_tf_tensor2 = tf.math.multiply(tf_tensor2, tf_tensor2, 1, 1)
   print(tf_math_multiply_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.PReLU api(*args)(tensor)
num = 1706759198.385153
try:
   tf_keras_layers_PReLU_0_tf_tensor2 = tf.keras.layers.PReLU()(tf_tensor2)
   print(tf_keras_layers_PReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.pow api(tensor+, *args)
num = 1706759198.3851562
try:
   tf_math_pow_0_tf_tensor2 = tf.math.pow(tf_tensor2, tf_tensor2, 1, 1)
   print(tf_math_pow_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.reciprocal api(tensor, *args)
num = 1706759198.385159
try:
   tf_math_reciprocal_0_tf_tensor2 = tf.math.reciprocal(tf_tensor2, 1)
   print(tf_math_reciprocal_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.reduce_max api(tensor, *args)
num = 1706759198.3851619
try:
   tf_reduce_max_0_tf_tensor2 = tf.reduce_max(tf_tensor2, 1)
   print(tf_reduce_max_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.reduce_mean api(tensor, *args)
num = 1706759198.385165
try:
   tf_reduce_mean_0_tf_tensor2 = tf.reduce_mean(tf_tensor2, 1)
   print(tf_reduce_mean_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.reduce_min api(tensor, *args)
num = 1706759198.385171
try:
   tf_reduce_min_0_tf_tensor2 = tf.reduce_min(tf_tensor2, 1)
   print(tf_reduce_min_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.reduce_prod api(tensor, *args)
num = 1706759198.385174
try:
   tf_reduce_prod_0_tf_tensor2 = tf.reduce_prod(tf_tensor2, 1)
   print(tf_reduce_prod_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.reduce_sum api(tensor, *args)
num = 1706759198.3851762
try:
   tf_reduce_sum_0_tf_tensor2 = tf.reduce_sum(tf_tensor2, 1)
   print(tf_reduce_sum_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ReLU api(*args)(tensor)
num = 1706759198.3851788
try:
   tf_keras_layers_ReLU_0_tf_tensor2 = tf.keras.layers.ReLU()(tf_tensor2)
   print(tf_keras_layers_ReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.nn.relu api(tensor, *args)
num = 1706759198.385181
try:
   tf_nn_relu_1_tf_tensor2 = tf.nn.relu(tf_tensor2, 1)
   print(tf_nn_relu_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.relu api(tensor, *args)
num = 1706759198.385184
try:
   tf_keras_activations_relu_2_tf_tensor2 = tf.keras.activations.relu(tf_tensor2, 1)
   print(tf_keras_activations_relu_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Reshape api(*args)(tensor)
num = 1706759198.3851871
try:
   tf_keras_layers_Reshape_0_tf_tensor2 = tf.keras.layers.Reshape(1)(tf_tensor2)
   print(tf_keras_layers_Reshape_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.round api(tensor, *args)
num = 1706759198.3851898
try:
   tf_math_round_0_tf_tensor2 = tf.math.round(tf_tensor2, 1)
   print(tf_math_round_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.selu api(tensor, *args)
num = 1706759198.385192
try:
   tf_keras_activations_selu_0_tf_tensor2 = tf.keras.activations.selu(tf_tensor2, 1)
   print(tf_keras_activations_selu_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.nn.selu api(tensor, *args)
num = 1706759198.385194
try:
   tf_nn_selu_1_tf_tensor2 = tf.nn.selu(tf_tensor2, 1)
   print(tf_nn_selu_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.sigmoid api(tensor, *args)
num = 1706759198.3851972
try:
   tf_keras_activations_sigmoid_0_tf_tensor2 = tf.keras.activations.sigmoid(tf_tensor2, 1)
   print(tf_keras_activations_sigmoid_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.sigmoid api(tensor, *args)
num = 1706759198.3852
try:
   tf_math_sigmoid_1_tf_tensor2 = tf.math.sigmoid(tf_tensor2, 1)
   print(tf_math_sigmoid_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Softmax api(*args)(tensor)
num = 1706759198.385202
try:
   tf_keras_layers_Softmax_0_tf_tensor2 = tf.keras.layers.Softmax()(tf_tensor2)
   print(tf_keras_layers_Softmax_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.softmax api(tensor, *args)
num = 1706759198.385205
try:
   tf_keras_activations_softmax_1_tf_tensor2 = tf.keras.activations.softmax(tf_tensor2, 1)
   print(tf_keras_activations_softmax_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.nn.softmax api(tensor, *args)
num = 1706759198.385207
try:
   tf_nn_softmax_2_tf_tensor2 = tf.nn.softmax(tf_tensor2, 1)
   print(tf_nn_softmax_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.softplus api(tensor, *args)
num = 1706759198.38521
try:
   tf_keras_activations_softplus_0_tf_tensor2 = tf.keras.activations.softplus(tf_tensor2, 1)
   print(tf_keras_activations_softplus_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.softplus api(tensor, *args)
num = 1706759198.385212
try:
   tf_math_softplus_1_tf_tensor2 = tf.math.softplus(tf_tensor2, 1)
   print(tf_math_softplus_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.softsign api(tensor, *args)
num = 1706759198.385215
try:
   tf_keras_activations_softsign_0_tf_tensor2 = tf.keras.activations.softsign(tf_tensor2, 1)
   print(tf_keras_activations_softsign_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.nn.softsign api(tensor, *args)
num = 1706759198.385217
try:
   tf_nn_softsign_1_tf_tensor2 = tf.nn.softsign(tf_tensor2, 1)
   print(tf_nn_softsign_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.subtract api(tensor+, *args)
num = 1706759198.38522
try:
   tf_math_subtract_0_tf_tensor2 = tf.math.subtract(tf_tensor2, tf_tensor2, 1, 1)
   print(tf_math_subtract_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.activations.tanh api(tensor, *args)
num = 1706759198.3852239
try:
   tf_keras_activations_tanh_0_tf_tensor2 = tf.keras.activations.tanh(tf_tensor2, 1)
   print(tf_keras_activations_tanh_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.math.tanh api(tensor, *args)
num = 1706759198.385226
try:
   tf_math_tanh_1_tf_tensor2 = tf.math.tanh(tf_tensor2, 1)
   print(tf_math_tanh_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ThresholdedReLU api(*args)(tensor)
num = 1706759198.385229
try:
   tf_keras_layers_ThresholdedReLU_0_tf_tensor2 = tf.keras.layers.ThresholdedReLU()(tf_tensor2)
   print(tf_keras_layers_ThresholdedReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

