import tensorflow as tf
import numpy as np
from tensorflow import keras
from qstransformer_layers import TransformerBlock, TokenAndPositionEmbedding, MultiHeadAttentionBlock


def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
	return pos * angle_rates


def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	pos_encoding = angle_rads[np.newaxis, ...]

	return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

	# add extra dimensions to add the padding
	# to the attention logits.
	return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

#Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    # self.mha = MultiHeadAttention(d_model, num_heads)
    # self.ffn = point_wise_feed_forward_network(d_model, dff)

    # self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    # self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # self.dropout1 = tf.keras.layers.Dropout(rate)
    # self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    # attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    # attn_output = self.dropout1(attn_output, training=training)
    # out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    # ffn_output = self.dropout2(ffn_output, training=training)
    # out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    # return out2
