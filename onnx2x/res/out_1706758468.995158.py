
import tensorflow as tf

tensor_pattern = [
    'api(tensor+)',
    'api(*args)(tensor)',
    'api(*args)([tensor+])',
    'api([tensor+], *args)',
    'api([tensor+])',
]

tf_tensor1 = tf.ones(shape=(1, 4), dtype=tf.float32)
tf_tensor2 = tf.ones(shape=(1, 1, 4), dtype=tf.float32)
tf_tensor3 = tf.ones(shape=(1, 1, 1, 4), dtype=tf.float32)

# tf.keras.layers.Add api(*args)([tensor+])
num = 1706758468.994921
try:
   tf_keras_layers_Add_0_tf_tensor2 = tf.keras.layers.Add()([tf_tensor2, tf_tensor2])
   print(tf_keras_layers_Add_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling2D api(*args)(tensor)
num = 1706758468.9949372
try:
   tf_keras_layers_AveragePooling2D_0_tf_tensor2 = tf.keras.layers.AveragePooling2D()(tf_tensor2)
   print(tf_keras_layers_AveragePooling2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling3D api(*args)(tensor)
num = 1706758468.994943
try:
   tf_keras_layers_AveragePooling3D_1_tf_tensor3 = tf.keras.layers.AveragePooling3D()(tf_tensor3)
   print(tf_keras_layers_AveragePooling3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling1D api(*args)(tensor)
num = 1706758468.994947
try:
   tf_keras_layers_AveragePooling1D_2_tf_tensor1 = tf.keras.layers.AveragePooling1D()(tf_tensor1)
   print(tf_keras_layers_AveragePooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.BatchNormalization api(*args)(tensor)
num = 1706758468.994952
try:
   tf_keras_layers_BatchNormalization_0_tf_tensor2 = tf.keras.layers.BatchNormalization()(tf_tensor2)
   print(tf_keras_layers_BatchNormalization_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.concat api([tensor+], *args)
num = 1706758468.9949608
try:
   tf_concat_0_tf_tensor2 = tf.concat([tf_tensor2, tf_tensor2], 1, 1)
   print(tf_concat_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Concatenate api(*args)([tensor+])
num = 1706758468.994964
try:
   tf_keras_layers_Concatenate_1_tf_tensor2 = tf.keras.layers.Concatenate()([tf_tensor2, tf_tensor2])
   print(tf_keras_layers_Concatenate_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2D api(*args)(tensor)
num = 1706758468.9949691
try:
   tf_keras_layers_Conv2D_0_tf_tensor2 = tf.keras.layers.Conv2D(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3D api(*args)(tensor)
num = 1706758468.994974
try:
   tf_keras_layers_Conv3D_1_tf_tensor3 = tf.keras.layers.Conv3D(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1D api(*args)(tensor)
num = 1706758468.994978
try:
   tf_keras_layers_Conv1D_2_tf_tensor1 = tf.keras.layers.Conv1D(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1DTranspose api(*args)(tensor)
num = 1706758468.994983
try:
   tf_keras_layers_Conv1DTranspose_0_tf_tensor1 = tf.keras.layers.Conv1DTranspose(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv1DTranspose_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3DTranspose api(*args)(tensor)
num = 1706758468.994987
try:
   tf_keras_layers_Conv3DTranspose_1_tf_tensor3 = tf.keras.layers.Conv3DTranspose(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv3DTranspose_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2DTranspose api(*args)(tensor)
num = 1706758468.994992
try:
   tf_keras_layers_Conv2DTranspose_2_tf_tensor2 = tf.keras.layers.Conv2DTranspose(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv2DTranspose_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Dropout api(*args)(tensor)
num = 1706758468.994996
try:
   tf_keras_layers_Dropout_0_tf_tensor2 = tf.keras.layers.Dropout(1)(tf_tensor2)
   print(tf_keras_layers_Dropout_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ELU api(*args)(tensor)
num = 1706758468.995002
try:
   tf_keras_layers_ELU_1_tf_tensor2 = tf.keras.layers.ELU()(tf_tensor2)
   print(tf_keras_layers_ELU_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Flatten api(*args)(tensor)
num = 1706758468.995011
try:
   tf_keras_layers_Flatten_0_tf_tensor2 = tf.keras.layers.Flatten()(tf_tensor2)
   print(tf_keras_layers_Flatten_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling2D api(*args)(tensor)
num = 1706758468.995023
try:
   tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor2 = tf.keras.layers.GlobalAveragePooling2D()(tf_tensor2)
   print(tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling3D api(*args)(tensor)
num = 1706758468.9950259
try:
   tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor3 = tf.keras.layers.GlobalAveragePooling3D()(tf_tensor3)
   print(tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling1D api(*args)(tensor)
num = 1706758468.995029
try:
   tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor1 = tf.keras.layers.GlobalAveragePooling1D()(tf_tensor1)
   print(tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling3D api(*args)(tensor)
num = 1706758468.995032
try:
   tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor3 = tf.keras.layers.GlobalMaxPooling3D()(tf_tensor3)
   print(tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling1D api(*args)(tensor)
num = 1706758468.995035
try:
   tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor1 = tf.keras.layers.GlobalMaxPooling1D()(tf_tensor1)
   print(tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling2D api(*args)(tensor)
num = 1706758468.9950368
try:
   tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor2 = tf.keras.layers.GlobalMaxPooling2D()(tf_tensor2)
   print(tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LayerNormalization api(*args)(tensor)
num = 1706758468.995046
try:
   tf_keras_layers_LayerNormalization_0_tf_tensor2 = tf.keras.layers.LayerNormalization()(tf_tensor2)
   print(tf_keras_layers_LayerNormalization_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LeakyReLU api(*args)(tensor)
num = 1706758468.995049
try:
   tf_keras_layers_LeakyReLU_0_tf_tensor2 = tf.keras.layers.LeakyReLU()(tf_tensor2)
   print(tf_keras_layers_LeakyReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling3D api(*args)(tensor)
num = 1706758468.99506
try:
   tf_keras_layers_MaxPooling3D_0_tf_tensor3 = tf.keras.layers.MaxPooling3D()(tf_tensor3)
   print(tf_keras_layers_MaxPooling3D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling2D api(*args)(tensor)
num = 1706758468.995063
try:
   tf_keras_layers_MaxPooling2D_1_tf_tensor2 = tf.keras.layers.MaxPooling2D()(tf_tensor2)
   print(tf_keras_layers_MaxPooling2D_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling1D api(*args)(tensor)
num = 1706758468.9950662
try:
   tf_keras_layers_MaxPooling1D_2_tf_tensor1 = tf.keras.layers.MaxPooling1D()(tf_tensor1)
   print(tf_keras_layers_MaxPooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.PReLU api(*args)(tensor)
num = 1706758468.995082
try:
   tf_keras_layers_PReLU_0_tf_tensor2 = tf.keras.layers.PReLU()(tf_tensor2)
   print(tf_keras_layers_PReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ReLU api(*args)(tensor)
num = 1706758468.995106
try:
   tf_keras_layers_ReLU_0_tf_tensor2 = tf.keras.layers.ReLU()(tf_tensor2)
   print(tf_keras_layers_ReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Reshape api(*args)(tensor)
num = 1706758468.995114
try:
   tf_keras_layers_Reshape_0_tf_tensor2 = tf.keras.layers.Reshape(1)(tf_tensor2)
   print(tf_keras_layers_Reshape_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Softmax api(*args)(tensor)
num = 1706758468.99513
try:
   tf_keras_layers_Softmax_0_tf_tensor2 = tf.keras.layers.Softmax()(tf_tensor2)
   print(tf_keras_layers_Softmax_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ThresholdedReLU api(*args)(tensor)
num = 1706758468.995157
try:
   tf_keras_layers_ThresholdedReLU_0_tf_tensor2 = tf.keras.layers.ThresholdedReLU()(tf_tensor2)
   print(tf_keras_layers_ThresholdedReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

