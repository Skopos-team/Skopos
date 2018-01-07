import tensorflow as tf
import numpy as np
from skopos.network.layers import Layer

class RecurrentLayer(Layer):
	"""docstring for RecurrentLayer"""
	def __init__(self):
		super(RecurrentLayer, self).__init__()

class LSTM(RecurrentLayer):
	"""docstring for LSTM"""
	def __init__(self, size=100):
		super(LSTM, self).__init__()
		self.size = size
		self.initialize_LSTM_cell()

	def initialize_LSTM_cell(self):
		with tf.variable_scope(self.get_scope(), reuse=False):
			"""
				Create a LSTM cell with UNITS_LSTM units. We then define an initial
				state for the LSTM cell. The initial state is divided into 'c' and 'h',
				each with dimension [LSTM_BATCH_SIZE, LSTM_CELL_STATE_SIZE], where
				LSTM_BATCH_SIZE=1. We use 'self.lstm_state_init' only when we want
				te reset the LSTM layer's state (between episodes, for example)
			"""
			self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.size, state_is_tuple=True)
			c_init = np.zeros((1, self.cell.state_size.c), np.float32)
			h_init = np.zeros((1, self.cell.state_size.h), np.float32)
			self.state_in = tf.contrib.rnn.LSTMStateTuple(c_init, h_init)
		return

	def apply_layer(self, x):
		"""
			Create new dimension, so that the input for the LSTM is (if using the
			fully connected layer):
			[batch_size=1, time_step=Conv_layer_batch_size, input_dim=UNITS_H1]
				- lstm_input.get_shape() ===> (1, ?, UNITS_H1)
			or, if not using the fully connected layer, it is:
			[batch_size=1, time_step=Conv_layer_batch_size, input_dim=SIZE_CONV_FLAT]
				- lstm_input.get_shape() ===> (1, ?, SIZE_CONV_FLAT)
		"""
		out = tf.expand_dims(x, [0])
		"""
			Retrives the original batch size (used in the Convolutional layer).
			This value (represented by the ? in 'lstm_input.get_shape()') indicates
			how many steps the LSTM layer will unroll. This makes sense, since the
			current batch carries an ordered sequence of events, which is required
			by the LSTM layer.
		"""
		step_size = tf.shape(x)[:1]
		""" 
			The update of the state is done dinamically, the next state is assigned to state_in, in order
			to avoid to feed the variable manually 
		"""
		self.rnn, self.state_in = tf.nn.dynamic_rnn(inputs=out, cell=self.cell, 
			dtype=tf.float32, initial_state=self.state_in, sequence_length=step_size)
		return tf.reshape(self.rnn, [-1, self.size])

	def get_scope(self):
		return "lstm"

class RNN(RecurrentLayer):
	""" Implementing RNN Layer"""
	def __init__(self, arg):
		super(RNN, self).__init__()
		self.arg = arg
		

		
