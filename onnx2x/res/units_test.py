# -*- coding: utf-8 -*-

"""
@Title   : 
@Time    : 2024/2/1 11:49
@Author  : Biophilia Wu
@Email   : BiophiliaSWDA@163.com
"""
import tensorflow as tf
tf_tensor1 = tf.ones(shape=(1, 4), dtype=tf.float32)
tf_tensor2 = tf.ones(shape=(1, 1, 4), dtype=tf.float32)
tf_tensor3 = tf.ones(shape=(1, 1, 1, 4), dtype=tf.float32)
tf_tensor4 = tf.ones(shape=(1, 1, 1, 1, 4), dtype=tf.float32)

tf_raw_ops_LRN_0_tf_tensor3 = tf.raw_ops.LRN(tf_tensor3, 1)
print(tf_raw_ops_LRN_0_tf_tensor3)
