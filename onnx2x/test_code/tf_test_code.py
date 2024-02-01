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
tf_tensor4 = tf.ones(shape=(1, 1, 1, 1, 4), dtype=tf.float32)

# tf.keras.layers.Add api(*args)([tensor+])
num = 1706795829.314462
try:
   tf_keras_layers_Add_0_tf_tensor3 = tf.keras.layers.Add()([tf_tensor3, tf_tensor3])
   print(tf_keras_layers_Add_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling2D api(*args)(tensor)
num = 1706795829.314477
try:
   tf_keras_layers_AveragePooling2D_0_tf_tensor3 = tf.keras.layers.AveragePooling2D()(tf_tensor3)
   print(tf_keras_layers_AveragePooling2D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling3D api(*args)(tensor)
num = 1706795829.314483
try:
   tf_keras_layers_AveragePooling3D_1_tf_tensor4 = tf.keras.layers.AveragePooling3D()(tf_tensor4)
   print(tf_keras_layers_AveragePooling3D_1_tf_tensor4)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling1D api(*args)(tensor)
num = 1706795829.314486
try:
   tf_keras_layers_AveragePooling1D_2_tf_tensor2 = tf.keras.layers.AveragePooling1D()(tf_tensor2)
   print(tf_keras_layers_AveragePooling1D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.BatchNormalization api(*args)(tensor)
num = 1706795829.3144908
try:
   tf_keras_layers_BatchNormalization_0_tf_tensor3 = tf.keras.layers.BatchNormalization()(tf_tensor3)
   print(tf_keras_layers_BatchNormalization_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.ceil api(tensor, *args)
num = 1706795829.314495
try:
   tf_math_ceil_0_tf_tensor3 = tf.math.ceil(tf_tensor3, )
   print(tf_math_ceil_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.concat api([tensor+], *args)
num = 1706795829.314502
try:
   tf_concat_0_tf_tensor3 = tf.concat([tf_tensor3, tf_tensor3], 1)
   print(tf_concat_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Concatenate api(*args)([tensor+])
num = 1706795829.3145049
try:
   tf_keras_layers_Concatenate_1_tf_tensor3 = tf.keras.layers.Concatenate()([tf_tensor3, tf_tensor3])
   print(tf_keras_layers_Concatenate_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2D api(*args)(tensor)
num = 1706795829.31451
try:
   tf_keras_layers_Conv2D_0_tf_tensor3 = tf.keras.layers.Conv2D(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv2D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3D api(*args)(tensor)
num = 1706795829.3145149
try:
   tf_keras_layers_Conv3D_1_tf_tensor4 = tf.keras.layers.Conv3D(1, 1)(tf_tensor4)
   print(tf_keras_layers_Conv3D_1_tf_tensor4)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1D api(*args)(tensor)
num = 1706795829.314519
try:
   tf_keras_layers_Conv1D_2_tf_tensor2 = tf.keras.layers.Conv1D(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv1D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1DTranspose api(*args)(tensor)
num = 1706795829.314524
try:
   tf_keras_layers_Conv1DTranspose_0_tf_tensor2 = tf.keras.layers.Conv1DTranspose(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv1DTranspose_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3DTranspose api(*args)(tensor)
num = 1706795829.314529
try:
   tf_keras_layers_Conv3DTranspose_1_tf_tensor4 = tf.keras.layers.Conv3DTranspose(1, 1)(tf_tensor4)
   print(tf_keras_layers_Conv3DTranspose_1_tf_tensor4)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2DTranspose api(*args)(tensor)
num = 1706795829.3145332
try:
   tf_keras_layers_Conv2DTranspose_2_tf_tensor3 = tf.keras.layers.Conv2DTranspose(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv2DTranspose_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Dropout api(*args)(tensor)
num = 1706795829.3145359
try:
   tf_keras_layers_Dropout_0_tf_tensor3 = tf.keras.layers.Dropout(1)(tf_tensor3)
   print(tf_keras_layers_Dropout_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.elu api(tensor, *args)
num = 1706795829.3145401
try:
   tf_keras_activations_elu_0_tf_tensor3 = tf.keras.activations.elu(tf_tensor3, )
   print(tf_keras_activations_elu_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ELU api(*args)(tensor)
num = 1706795829.314542
try:
   tf_keras_layers_ELU_1_tf_tensor3 = tf.keras.layers.ELU()(tf_tensor3)
   print(tf_keras_layers_ELU_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.nn.elu api(tensor, *args)
num = 1706795829.314545
try:
   tf_nn_elu_2_tf_tensor3 = tf.nn.elu(tf_tensor3, )
   print(tf_nn_elu_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.erf api(tensor, *args)
num = 1706795829.314548
try:
   tf_math_erf_0_tf_tensor3 = tf.math.erf(tf_tensor3, )
   print(tf_math_erf_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Flatten api(*args)(tensor)
num = 1706795829.314551
try:
   tf_keras_layers_Flatten_0_tf_tensor3 = tf.keras.layers.Flatten()(tf_tensor3)
   print(tf_keras_layers_Flatten_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.floor api(tensor, *args)
num = 1706795829.3145542
try:
   tf_math_floor_0_tf_tensor3 = tf.math.floor(tf_tensor3, )
   print(tf_math_floor_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.gelu api(tensor, *args)
num = 1706795829.3145568
try:
   tf_keras_activations_gelu_0_tf_tensor3 = tf.keras.activations.gelu(tf_tensor3, )
   print(tf_keras_activations_gelu_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.nn.gelu api(tensor, *args)
num = 1706795829.314559
try:
   tf_nn_gelu_1_tf_tensor3 = tf.nn.gelu(tf_tensor3, )
   print(tf_nn_gelu_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling2D api(*args)(tensor)
num = 1706795829.314563
try:
   tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor3 = tf.keras.layers.GlobalAveragePooling2D()(tf_tensor3)
   print(tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling3D api(*args)(tensor)
num = 1706795829.314566
try:
   tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor4 = tf.keras.layers.GlobalAveragePooling3D()(tf_tensor4)
   print(tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor4)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling1D api(*args)(tensor)
num = 1706795829.314569
try:
   tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor2 = tf.keras.layers.GlobalAveragePooling1D()(tf_tensor2)
   print(tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling3D api(*args)(tensor)
num = 1706795829.314572
try:
   tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor4 = tf.keras.layers.GlobalMaxPooling3D()(tf_tensor4)
   print(tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor4)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling1D api(*args)(tensor)
num = 1706795829.314574
try:
   tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor2 = tf.keras.layers.GlobalMaxPooling1D()(tf_tensor2)
   print(tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling2D api(*args)(tensor)
num = 1706795829.3145769
try:
   tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor3 = tf.keras.layers.GlobalMaxPooling2D()(tf_tensor3)
   print(tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.raw_ops.LRN api(tensor, *args)
num = 1706795829.314582
try:
   tf_raw_ops_LRN_0_tf_tensor3 = tf.raw_ops.LRN(tf_tensor3, )
   print(tf_raw_ops_LRN_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LayerNormalization api(*args)(tensor)
num = 1706795829.314585
try:
   tf_keras_layers_LayerNormalization_0_tf_tensor3 = tf.keras.layers.LayerNormalization()(tf_tensor3)
   print(tf_keras_layers_LayerNormalization_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LeakyReLU api(*args)(tensor)
num = 1706795829.3145878
try:
   tf_keras_layers_LeakyReLU_0_tf_tensor3 = tf.keras.layers.LeakyReLU()(tf_tensor3)
   print(tf_keras_layers_LeakyReLU_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.maximum api(tensor+, *args)
num = 1706795829.314592
try:
   tf_math_maximum_0_tf_tensor3 = tf.math.maximum(tf_tensor3, tf_tensor3, )
   print(tf_math_maximum_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.maximum api([tensor+], *args)
num = 1706795829.314595
try:
   tf_keras_layers_maximum_1_tf_tensor3 = tf.keras.layers.maximum([tf_tensor3, tf_tensor3], )
   print(tf_keras_layers_maximum_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling3D api(*args)(tensor)
num = 1706795829.3145988
try:
   tf_keras_layers_MaxPooling3D_0_tf_tensor4 = tf.keras.layers.MaxPooling3D()(tf_tensor4)
   print(tf_keras_layers_MaxPooling3D_0_tf_tensor4)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling2D api(*args)(tensor)
num = 1706795829.314602
try:
   tf_keras_layers_MaxPooling2D_1_tf_tensor3 = tf.keras.layers.MaxPooling2D()(tf_tensor3)
   print(tf_keras_layers_MaxPooling2D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling1D api(*args)(tensor)
num = 1706795829.314604
try:
   tf_keras_layers_MaxPooling1D_2_tf_tensor2 = tf.keras.layers.MaxPooling1D()(tf_tensor2)
   print(tf_keras_layers_MaxPooling1D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.minimum api(tensor+, *args)
num = 1706795829.314608
try:
   tf_minimum_0_tf_tensor3 = tf.minimum(tf_tensor3, tf_tensor3, )
   print(tf_minimum_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.mish api(tensor, *args)
num = 1706795829.314611
try:
   tf_keras_activations_mish_0_tf_tensor3 = tf.keras.activations.mish(tf_tensor3, )
   print(tf_keras_activations_mish_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.floormod api(tensor+, *args)
num = 1706795829.314614
try:
   tf_math_floormod_0_tf_tensor3 = tf.math.floormod(tf_tensor3, tf_tensor3, )
   print(tf_math_floormod_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.multiply api(tensor+, *args)
num = 1706795829.3146172
try:
   tf_math_multiply_0_tf_tensor3 = tf.math.multiply(tf_tensor3, tf_tensor3, )
   print(tf_math_multiply_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.PReLU api(*args)(tensor)
num = 1706795829.31462
try:
   tf_keras_layers_PReLU_0_tf_tensor3 = tf.keras.layers.PReLU()(tf_tensor3)
   print(tf_keras_layers_PReLU_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.pow api(tensor+, *args)
num = 1706795829.3146229
try:
   tf_math_pow_0_tf_tensor3 = tf.math.pow(tf_tensor3, tf_tensor3, )
   print(tf_math_pow_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.reciprocal api(tensor, *args)
num = 1706795829.314626
try:
   tf_math_reciprocal_0_tf_tensor3 = tf.math.reciprocal(tf_tensor3, )
   print(tf_math_reciprocal_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.reduce_max api(tensor, *args)
num = 1706795829.314629
try:
   tf_reduce_max_0_tf_tensor3 = tf.reduce_max(tf_tensor3, )
   print(tf_reduce_max_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.reduce_mean api(tensor, *args)
num = 1706795829.314633
try:
   tf_reduce_mean_0_tf_tensor3 = tf.reduce_mean(tf_tensor3, )
   print(tf_reduce_mean_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.reduce_min api(tensor, *args)
num = 1706795829.314636
try:
   tf_reduce_min_0_tf_tensor3 = tf.reduce_min(tf_tensor3, )
   print(tf_reduce_min_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.reduce_prod api(tensor, *args)
num = 1706795829.314639
try:
   tf_reduce_prod_0_tf_tensor3 = tf.reduce_prod(tf_tensor3, )
   print(tf_reduce_prod_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.reduce_sum api(tensor, *args)
num = 1706795829.314642
try:
   tf_reduce_sum_0_tf_tensor3 = tf.reduce_sum(tf_tensor3, )
   print(tf_reduce_sum_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ReLU api(*args)(tensor)
num = 1706795829.314645
try:
   tf_keras_layers_ReLU_0_tf_tensor3 = tf.keras.layers.ReLU()(tf_tensor3)
   print(tf_keras_layers_ReLU_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.nn.relu api(tensor, *args)
num = 1706795829.314647
try:
   tf_nn_relu_1_tf_tensor3 = tf.nn.relu(tf_tensor3, )
   print(tf_nn_relu_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.relu api(tensor, *args)
num = 1706795829.31465
try:
   tf_keras_activations_relu_2_tf_tensor3 = tf.keras.activations.relu(tf_tensor3, )
   print(tf_keras_activations_relu_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Reshape api(*args)(tensor)
num = 1706795829.314653
try:
   tf_keras_layers_Reshape_0_tf_tensor3 = tf.keras.layers.Reshape(1)(tf_tensor3)
   print(tf_keras_layers_Reshape_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.round api(tensor, *args)
num = 1706795829.314656
try:
   tf_math_round_0_tf_tensor3 = tf.math.round(tf_tensor3, )
   print(tf_math_round_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.selu api(tensor, *args)
num = 1706795829.314659
try:
   tf_keras_activations_selu_0_tf_tensor3 = tf.keras.activations.selu(tf_tensor3, )
   print(tf_keras_activations_selu_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.nn.selu api(tensor, *args)
num = 1706795829.314661
try:
   tf_nn_selu_1_tf_tensor3 = tf.nn.selu(tf_tensor3, )
   print(tf_nn_selu_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.sigmoid api(tensor, *args)
num = 1706795829.314664
try:
   tf_keras_activations_sigmoid_0_tf_tensor3 = tf.keras.activations.sigmoid(tf_tensor3, )
   print(tf_keras_activations_sigmoid_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.sigmoid api(tensor, *args)
num = 1706795829.314666
try:
   tf_math_sigmoid_1_tf_tensor3 = tf.math.sigmoid(tf_tensor3, )
   print(tf_math_sigmoid_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Softmax api(*args)(tensor)
num = 1706795829.3146691
try:
   tf_keras_layers_Softmax_0_tf_tensor3 = tf.keras.layers.Softmax()(tf_tensor3)
   print(tf_keras_layers_Softmax_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.softmax api(tensor, *args)
num = 1706795829.314671
try:
   tf_keras_activations_softmax_1_tf_tensor3 = tf.keras.activations.softmax(tf_tensor3, )
   print(tf_keras_activations_softmax_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.nn.softmax api(tensor, *args)
num = 1706795829.314674
try:
   tf_nn_softmax_2_tf_tensor3 = tf.nn.softmax(tf_tensor3, )
   print(tf_nn_softmax_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.softplus api(tensor, *args)
num = 1706795829.314677
try:
   tf_keras_activations_softplus_0_tf_tensor3 = tf.keras.activations.softplus(tf_tensor3, )
   print(tf_keras_activations_softplus_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.softplus api(tensor, *args)
num = 1706795829.3146791
try:
   tf_math_softplus_1_tf_tensor3 = tf.math.softplus(tf_tensor3, )
   print(tf_math_softplus_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.softsign api(tensor, *args)
num = 1706795829.314682
try:
   tf_keras_activations_softsign_0_tf_tensor3 = tf.keras.activations.softsign(tf_tensor3, )
   print(tf_keras_activations_softsign_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.nn.softsign api(tensor, *args)
num = 1706795829.314684
try:
   tf_nn_softsign_1_tf_tensor3 = tf.nn.softsign(tf_tensor3, )
   print(tf_nn_softsign_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.subtract api(tensor+, *args)
num = 1706795829.314687
try:
   tf_math_subtract_0_tf_tensor3 = tf.math.subtract(tf_tensor3, tf_tensor3, )
   print(tf_math_subtract_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.activations.tanh api(tensor, *args)
num = 1706795829.31469
try:
   tf_keras_activations_tanh_0_tf_tensor3 = tf.keras.activations.tanh(tf_tensor3, )
   print(tf_keras_activations_tanh_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.math.tanh api(tensor, *args)
num = 1706795829.314692
try:
   tf_math_tanh_1_tf_tensor3 = tf.math.tanh(tf_tensor3, )
   print(tf_math_tanh_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ThresholdedReLU api(*args)(tensor)
num = 1706795829.314695
try:
   tf_keras_layers_ThresholdedReLU_0_tf_tensor3 = tf.keras.layers.ThresholdedReLU()(tf_tensor3)
   print(tf_keras_layers_ThresholdedReLU_0_tf_tensor3)
except Exception as e: 
   print(num, e)


