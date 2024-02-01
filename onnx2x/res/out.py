
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
num = 1706517305.010807
try:
   tf_keras_layers_Add_0_tf_tensor1 = tf.keras.layers.Add()([tf_tensor1, tf_tensor1])
   print(tf_keras_layers_Add_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Add api(*args)([tensor+])
num = 1706517305.010829
try:
   tf_keras_layers_Add_0_tf_tensor2 = tf.keras.layers.Add()([tf_tensor2, tf_tensor2])
   print(tf_keras_layers_Add_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Add api(*args)([tensor+])
num = 1706517305.0108318
try:
   tf_keras_layers_Add_0_tf_tensor3 = tf.keras.layers.Add()([tf_tensor3, tf_tensor3])
   print(tf_keras_layers_Add_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling2D api(*args)(tensor)
num = 1706517305.010839
try:
   tf_keras_layers_AveragePooling2D_0_tf_tensor1 = tf.keras.layers.AveragePooling2D()(tf_tensor1)
   print(tf_keras_layers_AveragePooling2D_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling2D api(*args)(tensor)
num = 1706517305.0108411
try:
   tf_keras_layers_AveragePooling2D_0_tf_tensor2 = tf.keras.layers.AveragePooling2D()(tf_tensor2)
   print(tf_keras_layers_AveragePooling2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling2D api(*args)(tensor)
num = 1706517305.010842
try:
   tf_keras_layers_AveragePooling2D_0_tf_tensor3 = tf.keras.layers.AveragePooling2D()(tf_tensor3)
   print(tf_keras_layers_AveragePooling2D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling3D api(*args)(tensor)
num = 1706517305.010845
try:
   tf_keras_layers_AveragePooling3D_1_tf_tensor1 = tf.keras.layers.AveragePooling3D()(tf_tensor1)
   print(tf_keras_layers_AveragePooling3D_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling3D api(*args)(tensor)
num = 1706517305.010846
try:
   tf_keras_layers_AveragePooling3D_1_tf_tensor2 = tf.keras.layers.AveragePooling3D()(tf_tensor2)
   print(tf_keras_layers_AveragePooling3D_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling3D api(*args)(tensor)
num = 1706517305.010847
try:
   tf_keras_layers_AveragePooling3D_1_tf_tensor3 = tf.keras.layers.AveragePooling3D()(tf_tensor3)
   print(tf_keras_layers_AveragePooling3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling1D api(*args)(tensor)
num = 1706517305.010881
try:
   tf_keras_layers_AveragePooling1D_2_tf_tensor1 = tf.keras.layers.AveragePooling1D()(tf_tensor1)
   print(tf_keras_layers_AveragePooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling1D api(*args)(tensor)
num = 1706517305.010886
try:
   tf_keras_layers_AveragePooling1D_2_tf_tensor2 = tf.keras.layers.AveragePooling1D()(tf_tensor2)
   print(tf_keras_layers_AveragePooling1D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.AveragePooling1D api(*args)(tensor)
num = 1706517305.0108879
try:
   tf_keras_layers_AveragePooling1D_2_tf_tensor3 = tf.keras.layers.AveragePooling1D()(tf_tensor3)
   print(tf_keras_layers_AveragePooling1D_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.BatchNormalization api(*args)(tensor)
num = 1706517305.010896
try:
   tf_keras_layers_BatchNormalization_0_tf_tensor1 = tf.keras.layers.BatchNormalization()(tf_tensor1)
   print(tf_keras_layers_BatchNormalization_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.BatchNormalization api(*args)(tensor)
num = 1706517305.0108979
try:
   tf_keras_layers_BatchNormalization_0_tf_tensor2 = tf.keras.layers.BatchNormalization()(tf_tensor2)
   print(tf_keras_layers_BatchNormalization_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.BatchNormalization api(*args)(tensor)
num = 1706517305.0108988
try:
   tf_keras_layers_BatchNormalization_0_tf_tensor3 = tf.keras.layers.BatchNormalization()(tf_tensor3)
   print(tf_keras_layers_BatchNormalization_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.concat api([tensor+], *args)
num = 1706517305.0109131
try:
   tf_concat_0_tf_tensor1 = tf.concat([tf_tensor1, tf_tensor1], 1, 1)
   print(tf_concat_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.concat api([tensor+], *args)
num = 1706517305.010915
try:
   tf_concat_0_tf_tensor2 = tf.concat([tf_tensor2, tf_tensor2], 1, 1)
   print(tf_concat_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.concat api([tensor+], *args)
num = 1706517305.010916
try:
   tf_concat_0_tf_tensor3 = tf.concat([tf_tensor3, tf_tensor3], 1, 1)
   print(tf_concat_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Concatenate api(*args)([tensor+])
num = 1706517305.0109189
try:
   tf_keras_layers_Concatenate_1_tf_tensor1 = tf.keras.layers.Concatenate()([tf_tensor1, tf_tensor1])
   print(tf_keras_layers_Concatenate_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Concatenate api(*args)([tensor+])
num = 1706517305.01092
try:
   tf_keras_layers_Concatenate_1_tf_tensor2 = tf.keras.layers.Concatenate()([tf_tensor2, tf_tensor2])
   print(tf_keras_layers_Concatenate_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Concatenate api(*args)([tensor+])
num = 1706517305.010922
try:
   tf_keras_layers_Concatenate_1_tf_tensor3 = tf.keras.layers.Concatenate()([tf_tensor3, tf_tensor3])
   print(tf_keras_layers_Concatenate_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2D api(*args)(tensor)
num = 1706517305.010926
try:
   tf_keras_layers_Conv2D_0_tf_tensor1 = tf.keras.layers.Conv2D(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv2D_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2D api(*args)(tensor)
num = 1706517305.0109282
try:
   tf_keras_layers_Conv2D_0_tf_tensor2 = tf.keras.layers.Conv2D(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2D api(*args)(tensor)
num = 1706517305.0109289
try:
   tf_keras_layers_Conv2D_0_tf_tensor3 = tf.keras.layers.Conv2D(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv2D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3D api(*args)(tensor)
num = 1706517305.010932
try:
   tf_keras_layers_Conv3D_1_tf_tensor1 = tf.keras.layers.Conv3D(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv3D_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3D api(*args)(tensor)
num = 1706517305.010934
try:
   tf_keras_layers_Conv3D_1_tf_tensor2 = tf.keras.layers.Conv3D(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv3D_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3D api(*args)(tensor)
num = 1706517305.010935
try:
   tf_keras_layers_Conv3D_1_tf_tensor3 = tf.keras.layers.Conv3D(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1D api(*args)(tensor)
num = 1706517305.0109382
try:
   tf_keras_layers_Conv1D_2_tf_tensor1 = tf.keras.layers.Conv1D(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1D api(*args)(tensor)
num = 1706517305.010939
try:
   tf_keras_layers_Conv1D_2_tf_tensor2 = tf.keras.layers.Conv1D(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv1D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1D api(*args)(tensor)
num = 1706517305.0109398
try:
   tf_keras_layers_Conv1D_2_tf_tensor3 = tf.keras.layers.Conv1D(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv1D_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1DTranspose api(*args)(tensor)
num = 1706517305.010945
try:
   tf_keras_layers_Conv1DTranspose_0_tf_tensor1 = tf.keras.layers.Conv1DTranspose(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv1DTranspose_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1DTranspose api(*args)(tensor)
num = 1706517305.010946
try:
   tf_keras_layers_Conv1DTranspose_0_tf_tensor2 = tf.keras.layers.Conv1DTranspose(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv1DTranspose_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv1DTranspose api(*args)(tensor)
num = 1706517305.010947
try:
   tf_keras_layers_Conv1DTranspose_0_tf_tensor3 = tf.keras.layers.Conv1DTranspose(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv1DTranspose_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3DTranspose api(*args)(tensor)
num = 1706517305.0109499
try:
   tf_keras_layers_Conv3DTranspose_1_tf_tensor1 = tf.keras.layers.Conv3DTranspose(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv3DTranspose_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3DTranspose api(*args)(tensor)
num = 1706517305.010952
try:
   tf_keras_layers_Conv3DTranspose_1_tf_tensor2 = tf.keras.layers.Conv3DTranspose(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv3DTranspose_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv3DTranspose api(*args)(tensor)
num = 1706517305.010953
try:
   tf_keras_layers_Conv3DTranspose_1_tf_tensor3 = tf.keras.layers.Conv3DTranspose(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv3DTranspose_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2DTranspose api(*args)(tensor)
num = 1706517305.010956
try:
   tf_keras_layers_Conv2DTranspose_2_tf_tensor1 = tf.keras.layers.Conv2DTranspose(1, 1)(tf_tensor1)
   print(tf_keras_layers_Conv2DTranspose_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2DTranspose api(*args)(tensor)
num = 1706517305.010957
try:
   tf_keras_layers_Conv2DTranspose_2_tf_tensor2 = tf.keras.layers.Conv2DTranspose(1, 1)(tf_tensor2)
   print(tf_keras_layers_Conv2DTranspose_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Conv2DTranspose api(*args)(tensor)
num = 1706517305.0109582
try:
   tf_keras_layers_Conv2DTranspose_2_tf_tensor3 = tf.keras.layers.Conv2DTranspose(1, 1)(tf_tensor3)
   print(tf_keras_layers_Conv2DTranspose_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Dropout api(*args)(tensor)
num = 1706517305.0109608
try:
   tf_keras_layers_Dropout_0_tf_tensor1 = tf.keras.layers.Dropout(1)(tf_tensor1)
   print(tf_keras_layers_Dropout_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Dropout api(*args)(tensor)
num = 1706517305.010962
try:
   tf_keras_layers_Dropout_0_tf_tensor2 = tf.keras.layers.Dropout(1)(tf_tensor2)
   print(tf_keras_layers_Dropout_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Dropout api(*args)(tensor)
num = 1706517305.010964
try:
   tf_keras_layers_Dropout_0_tf_tensor3 = tf.keras.layers.Dropout(1)(tf_tensor3)
   print(tf_keras_layers_Dropout_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ELU api(*args)(tensor)
num = 1706517305.01097
try:
   tf_keras_layers_ELU_1_tf_tensor1 = tf.keras.layers.ELU()(tf_tensor1)
   print(tf_keras_layers_ELU_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ELU api(*args)(tensor)
num = 1706517305.010972
try:
   tf_keras_layers_ELU_1_tf_tensor2 = tf.keras.layers.ELU()(tf_tensor2)
   print(tf_keras_layers_ELU_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ELU api(*args)(tensor)
num = 1706517305.010973
try:
   tf_keras_layers_ELU_1_tf_tensor3 = tf.keras.layers.ELU()(tf_tensor3)
   print(tf_keras_layers_ELU_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Flatten api(*args)(tensor)
num = 1706517305.010984
try:
   tf_keras_layers_Flatten_0_tf_tensor1 = tf.keras.layers.Flatten()(tf_tensor1)
   print(tf_keras_layers_Flatten_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Flatten api(*args)(tensor)
num = 1706517305.010985
try:
   tf_keras_layers_Flatten_0_tf_tensor2 = tf.keras.layers.Flatten()(tf_tensor2)
   print(tf_keras_layers_Flatten_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Flatten api(*args)(tensor)
num = 1706517305.010986
try:
   tf_keras_layers_Flatten_0_tf_tensor3 = tf.keras.layers.Flatten()(tf_tensor3)
   print(tf_keras_layers_Flatten_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling2D api(*args)(tensor)
num = 1706517305.0110028
try:
   tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor1 = tf.keras.layers.GlobalAveragePooling2D()(tf_tensor1)
   print(tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling2D api(*args)(tensor)
num = 1706517305.011004
try:
   tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor2 = tf.keras.layers.GlobalAveragePooling2D()(tf_tensor2)
   print(tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling2D api(*args)(tensor)
num = 1706517305.011005
try:
   tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor3 = tf.keras.layers.GlobalAveragePooling2D()(tf_tensor3)
   print(tf_keras_layers_GlobalAveragePooling2D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling3D api(*args)(tensor)
num = 1706517305.011007
try:
   tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor1 = tf.keras.layers.GlobalAveragePooling3D()(tf_tensor1)
   print(tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling3D api(*args)(tensor)
num = 1706517305.011008
try:
   tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor2 = tf.keras.layers.GlobalAveragePooling3D()(tf_tensor2)
   print(tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling3D api(*args)(tensor)
num = 1706517305.01101
try:
   tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor3 = tf.keras.layers.GlobalAveragePooling3D()(tf_tensor3)
   print(tf_keras_layers_GlobalAveragePooling3D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling1D api(*args)(tensor)
num = 1706517305.011012
try:
   tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor1 = tf.keras.layers.GlobalAveragePooling1D()(tf_tensor1)
   print(tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling1D api(*args)(tensor)
num = 1706517305.011013
try:
   tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor2 = tf.keras.layers.GlobalAveragePooling1D()(tf_tensor2)
   print(tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalAveragePooling1D api(*args)(tensor)
num = 1706517305.011014
try:
   tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor3 = tf.keras.layers.GlobalAveragePooling1D()(tf_tensor3)
   print(tf_keras_layers_GlobalAveragePooling1D_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling3D api(*args)(tensor)
num = 1706517305.011016
try:
   tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor1 = tf.keras.layers.GlobalMaxPooling3D()(tf_tensor1)
   print(tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling3D api(*args)(tensor)
num = 1706517305.011018
try:
   tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor2 = tf.keras.layers.GlobalMaxPooling3D()(tf_tensor2)
   print(tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling3D api(*args)(tensor)
num = 1706517305.011019
try:
   tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor3 = tf.keras.layers.GlobalMaxPooling3D()(tf_tensor3)
   print(tf_keras_layers_GlobalMaxPooling3D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling1D api(*args)(tensor)
num = 1706517305.0110211
try:
   tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor1 = tf.keras.layers.GlobalMaxPooling1D()(tf_tensor1)
   print(tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling1D api(*args)(tensor)
num = 1706517305.011022
try:
   tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor2 = tf.keras.layers.GlobalMaxPooling1D()(tf_tensor2)
   print(tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling1D api(*args)(tensor)
num = 1706517305.011023
try:
   tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor3 = tf.keras.layers.GlobalMaxPooling1D()(tf_tensor3)
   print(tf_keras_layers_GlobalMaxPooling1D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling2D api(*args)(tensor)
num = 1706517305.0110238
try:
   tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor1 = tf.keras.layers.GlobalMaxPooling2D()(tf_tensor1)
   print(tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling2D api(*args)(tensor)
num = 1706517305.011026
try:
   tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor2 = tf.keras.layers.GlobalMaxPooling2D()(tf_tensor2)
   print(tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.GlobalMaxPooling2D api(*args)(tensor)
num = 1706517305.0110269
try:
   tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor3 = tf.keras.layers.GlobalMaxPooling2D()(tf_tensor3)
   print(tf_keras_layers_GlobalMaxPooling2D_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LayerNormalization api(*args)(tensor)
num = 1706517305.0110369
try:
   tf_keras_layers_LayerNormalization_0_tf_tensor1 = tf.keras.layers.LayerNormalization()(tf_tensor1)
   print(tf_keras_layers_LayerNormalization_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LayerNormalization api(*args)(tensor)
num = 1706517305.011038
try:
   tf_keras_layers_LayerNormalization_0_tf_tensor2 = tf.keras.layers.LayerNormalization()(tf_tensor2)
   print(tf_keras_layers_LayerNormalization_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LayerNormalization api(*args)(tensor)
num = 1706517305.011039
try:
   tf_keras_layers_LayerNormalization_0_tf_tensor3 = tf.keras.layers.LayerNormalization()(tf_tensor3)
   print(tf_keras_layers_LayerNormalization_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LeakyReLU api(*args)(tensor)
num = 1706517305.011042
try:
   tf_keras_layers_LeakyReLU_0_tf_tensor1 = tf.keras.layers.LeakyReLU()(tf_tensor1)
   print(tf_keras_layers_LeakyReLU_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LeakyReLU api(*args)(tensor)
num = 1706517305.011043
try:
   tf_keras_layers_LeakyReLU_0_tf_tensor2 = tf.keras.layers.LeakyReLU()(tf_tensor2)
   print(tf_keras_layers_LeakyReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.LeakyReLU api(*args)(tensor)
num = 1706517305.011044
try:
   tf_keras_layers_LeakyReLU_0_tf_tensor3 = tf.keras.layers.LeakyReLU()(tf_tensor3)
   print(tf_keras_layers_LeakyReLU_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling3D api(*args)(tensor)
num = 1706517305.011057
try:
   tf_keras_layers_MaxPooling3D_0_tf_tensor1 = tf.keras.layers.MaxPooling3D()(tf_tensor1)
   print(tf_keras_layers_MaxPooling3D_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling3D api(*args)(tensor)
num = 1706517305.011059
try:
   tf_keras_layers_MaxPooling3D_0_tf_tensor2 = tf.keras.layers.MaxPooling3D()(tf_tensor2)
   print(tf_keras_layers_MaxPooling3D_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling3D api(*args)(tensor)
num = 1706517305.01106
try:
   tf_keras_layers_MaxPooling3D_0_tf_tensor3 = tf.keras.layers.MaxPooling3D()(tf_tensor3)
   print(tf_keras_layers_MaxPooling3D_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling2D api(*args)(tensor)
num = 1706517305.011062
try:
   tf_keras_layers_MaxPooling2D_1_tf_tensor1 = tf.keras.layers.MaxPooling2D()(tf_tensor1)
   print(tf_keras_layers_MaxPooling2D_1_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling2D api(*args)(tensor)
num = 1706517305.011063
try:
   tf_keras_layers_MaxPooling2D_1_tf_tensor2 = tf.keras.layers.MaxPooling2D()(tf_tensor2)
   print(tf_keras_layers_MaxPooling2D_1_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling2D api(*args)(tensor)
num = 1706517305.011064
try:
   tf_keras_layers_MaxPooling2D_1_tf_tensor3 = tf.keras.layers.MaxPooling2D()(tf_tensor3)
   print(tf_keras_layers_MaxPooling2D_1_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling1D api(*args)(tensor)
num = 1706517305.0110662
try:
   tf_keras_layers_MaxPooling1D_2_tf_tensor1 = tf.keras.layers.MaxPooling1D()(tf_tensor1)
   print(tf_keras_layers_MaxPooling1D_2_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling1D api(*args)(tensor)
num = 1706517305.011067
try:
   tf_keras_layers_MaxPooling1D_2_tf_tensor2 = tf.keras.layers.MaxPooling1D()(tf_tensor2)
   print(tf_keras_layers_MaxPooling1D_2_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.MaxPooling1D api(*args)(tensor)
num = 1706517305.0110679
try:
   tf_keras_layers_MaxPooling1D_2_tf_tensor3 = tf.keras.layers.MaxPooling1D()(tf_tensor3)
   print(tf_keras_layers_MaxPooling1D_2_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.PReLU api(*args)(tensor)
num = 1706517305.0110898
try:
   tf_keras_layers_PReLU_0_tf_tensor1 = tf.keras.layers.PReLU()(tf_tensor1)
   print(tf_keras_layers_PReLU_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.PReLU api(*args)(tensor)
num = 1706517305.011092
try:
   tf_keras_layers_PReLU_0_tf_tensor2 = tf.keras.layers.PReLU()(tf_tensor2)
   print(tf_keras_layers_PReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.PReLU api(*args)(tensor)
num = 1706517305.011093
try:
   tf_keras_layers_PReLU_0_tf_tensor3 = tf.keras.layers.PReLU()(tf_tensor3)
   print(tf_keras_layers_PReLU_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ReLU api(*args)(tensor)
num = 1706517305.01113
try:
   tf_keras_layers_ReLU_0_tf_tensor1 = tf.keras.layers.ReLU()(tf_tensor1)
   print(tf_keras_layers_ReLU_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ReLU api(*args)(tensor)
num = 1706517305.0111308
try:
   tf_keras_layers_ReLU_0_tf_tensor2 = tf.keras.layers.ReLU()(tf_tensor2)
   print(tf_keras_layers_ReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ReLU api(*args)(tensor)
num = 1706517305.011132
try:
   tf_keras_layers_ReLU_0_tf_tensor3 = tf.keras.layers.ReLU()(tf_tensor3)
   print(tf_keras_layers_ReLU_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Reshape api(*args)(tensor)
num = 1706517305.011143
try:
   tf_keras_layers_Reshape_0_tf_tensor1 = tf.keras.layers.Reshape(1)(tf_tensor1)
   print(tf_keras_layers_Reshape_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Reshape api(*args)(tensor)
num = 1706517305.011144
try:
   tf_keras_layers_Reshape_0_tf_tensor2 = tf.keras.layers.Reshape(1)(tf_tensor2)
   print(tf_keras_layers_Reshape_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Reshape api(*args)(tensor)
num = 1706517305.0111449
try:
   tf_keras_layers_Reshape_0_tf_tensor3 = tf.keras.layers.Reshape(1)(tf_tensor3)
   print(tf_keras_layers_Reshape_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Softmax api(*args)(tensor)
num = 1706517305.011169
try:
   tf_keras_layers_Softmax_0_tf_tensor1 = tf.keras.layers.Softmax()(tf_tensor1)
   print(tf_keras_layers_Softmax_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Softmax api(*args)(tensor)
num = 1706517305.0111701
try:
   tf_keras_layers_Softmax_0_tf_tensor2 = tf.keras.layers.Softmax()(tf_tensor2)
   print(tf_keras_layers_Softmax_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.Softmax api(*args)(tensor)
num = 1706517305.011171
try:
   tf_keras_layers_Softmax_0_tf_tensor3 = tf.keras.layers.Softmax()(tf_tensor3)
   print(tf_keras_layers_Softmax_0_tf_tensor3)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ThresholdedReLU api(*args)(tensor)
num = 1706517305.0112169
try:
   tf_keras_layers_ThresholdedReLU_0_tf_tensor1 = tf.keras.layers.ThresholdedReLU()(tf_tensor1)
   print(tf_keras_layers_ThresholdedReLU_0_tf_tensor1)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ThresholdedReLU api(*args)(tensor)
num = 1706517305.0112178
try:
   tf_keras_layers_ThresholdedReLU_0_tf_tensor2 = tf.keras.layers.ThresholdedReLU()(tf_tensor2)
   print(tf_keras_layers_ThresholdedReLU_0_tf_tensor2)
except Exception as e: 
   print(num, e)

# tf.keras.layers.ThresholdedReLU api(*args)(tensor)
num = 1706517305.01122
try:
   tf_keras_layers_ThresholdedReLU_0_tf_tensor3 = tf.keras.layers.ThresholdedReLU()(tf_tensor3)
   print(tf_keras_layers_ThresholdedReLU_0_tf_tensor3)
except Exception as e: 
   print(num, e)

