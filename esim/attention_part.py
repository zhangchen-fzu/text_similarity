
import keras.backend as K
import tensorflow as tf

class SoftAttention(object):

    def __call__(self, inputs):
        a = inputs[0] #batch * 8 * 128
        b = inputs[1] #batch * 16 * 128

        attention = tf.keras.layers.Lambda(self._attention,
                                        output_shape = self._attention_output_shape,
                                        arguments = None)(inputs) #batch * 8 * 16


        align_a = tf.keras.layers.Lambda(self._soft_alignment, #[batch * 8 * 16] * [batch * 16 * 128] = [batch* 8 * 128]
                                     output_shape = self._soft_alignment_output_shape,
                                     arguments = None)([attention, b])

        attention = K.permute_dimensions(attention, (0, 2, 1))
        align_b = tf.keras.layers.Lambda(self._soft_alignment, #[batch * 16 * 8] * [batch * 8 * 128] = [batch* 16 * 128]
                                     output_shape = self._soft_alignment_output_shape,
                                     arguments = None)([attention, a])

        return align_a, align_b

    def _attention(self, inputs):

        attn_weights = K.batch_dot(x=inputs[0],
                                   y=K.permute_dimensions(inputs[1], pattern=(0, 2, 1)))
        return attn_weights #batch * 8 * 16

    def _attention_output_shape(self, inputs):
        input_shape = inputs[0]
        embedding_size = input_shape[1]
        return (input_shape[0], embedding_size, embedding_size)

    def _soft_alignment(self, inputs):
        attention = inputs[0]
        sentence = inputs[1]

        # Subtract the max. from the attention weights to avoid overflows.
        exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
        exp_sum = K.sum(exp, axis=-1, keepdims=True)
        softmax = exp / exp_sum

        return K.batch_dot(softmax, sentence)

    def _soft_alignment_output_shape(self, inputs):
        attention_shape = inputs[0]
        sentence_shape = inputs[1]
        return (attention_shape[0], attention_shape[1], sentence_shape[2])