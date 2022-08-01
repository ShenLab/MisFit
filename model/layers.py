import typing
import warnings
import tensorflow as tf

class MultiHeadAttentionEdge(tf.keras.layers.Layer):

	def __init__(
		self,
		head_size: int,
		num_heads: int,
		output_size: int = None,
		dropout: float = 0.0,
		use_projection_bias: bool = True,
		kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
		kernel_regularizer: typing.Union[str, typing.Callable] = None,
		kernel_constraint: typing.Union[str, typing.Callable] = None,
		bias_initializer: typing.Union[str, typing.Callable] = "zeros",
		bias_regularizer: typing.Union[str, typing.Callable] = None,
		bias_constraint: typing.Union[str, typing.Callable] = None,
		**kwargs,
	):

		super().__init__(**kwargs)

		if output_size is not None and output_size < 1:
			raise ValueError("output_size must be a positive number")

		self.head_size = head_size
		self.num_heads = num_heads
		self.output_size = output_size
		self.use_projection_bias = use_projection_bias

		self.query_size = None
		self.key_size = None
		self.value_size = None
		self.edge_size = None

		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self.bias_constraint = tf.keras.constraints.get(bias_constraint)

		self.softmax = tf.keras.layers.Softmax()

		self.dropout = tf.keras.layers.Dropout(dropout)
		self._droput_rate = dropout

	def build(self, input_shape): #(query, key, value, edge)

		num_query_features = input_shape[0][-1]
		num_key_features = input_shape[1][-1]
		num_value_features = input_shape[2][-1]
		if len(input_shape) < 4:
			num_edge_features = 0
		else:
			num_edge_features = input_shape[3][-1]

		self.query_size = num_query_features
		self.key_size = num_key_features
		self.value_size = num_value_features
		self.edge_size = num_edge_features
		if self.output_size is None:
			self.output_size = num_value_features

		self.query_kernel = self.add_weight(
			name="query_kernel",
			shape=[self.num_heads, num_query_features, self.head_size],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
		)
		self.key_kernel = self.add_weight(
			name="key_kernel",
			shape=[self.num_heads, num_key_features + num_edge_features, self.head_size],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
		)
		self.value_kernel = self.add_weight(
			name="value_kernel",
			shape=[self.num_heads, num_value_features, self.head_size],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
		)
		self.projection_kernel = self.add_weight(
			name="projection_kernel",
			shape=[self.num_heads, self.head_size, self.output_size],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
		)

		if self.use_projection_bias:
			self.projection_bias = self.add_weight(
				name="projection_bias",
				shape=[self.output_size],
				initializer=self.bias_initializer,
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint,
			)
		else:
			self.projection_bias = None

		super().build(input_shape)

	def call(self, inputs, mask = None, return_attn_coef = False, training = False): # query, key, value, edge

		# einsum nomenclature
		# ------------------------
		# N = query elements
		# M = key/value elements
		# H = heads
		# I = input features
		# O = output features

		query = inputs[0]
		key = inputs[1]
		value = inputs[2]
		if len(inputs) < 4:
			edge = None
		else:
			edge = inputs[3]

		# verify shapes
		if tf.shape(key)[-2] != tf.shape(value)[-2]:
			raise ValueError(
				"the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
			)
		if edge is None:
			pass
		elif (tf.shape(key)[-2] != tf.shape(edge)[-2]) or (tf.shape(query)[-2] != tf.shape(edge)[-3]):
			raise ValueError(
				"the 'edge' second last dimension must be equal to 'key' second last dimension, the 'edge' third last dimension must be equal to 'query' second last dimension "
			)

		if mask is not None:
			if len(tf.shape(mask)) < 2:
				raise ValueError("'mask' must have at least 2 dimensions")
			if tf.shape(query)[-2] != tf.shape(mask)[-2]:
				raise ValueError(
					"mask's second to last dimension must be equal to the number of elements in 'query'"
				)
			if tf.shape(key)[-2] != tf.shape(mask)[-1]:
				raise ValueError(
					"mask's last dimension must be equal to the number of elements in 'key'"
				)

		# Linear transformations
		if edge is None:
			#key = tf.repeat(tf.expand_dims(key, -3), tf.shape(query)[-2], axis = -3)
			key = tf.einsum("...MI, HIO -> ...MHO", key, self.key_kernel)
		else:
			key = tf.concat([tf.repeat(tf.expand_dims(key, -3), tf.shape(query)[-2], axis = -3), edge], axis = -1)
			key = tf.einsum("...NMI, HIO -> ...NMHO", key, self.key_kernel)
		query = tf.einsum("...NI, HIO -> ...NHO", query, self.query_kernel)
		value = tf.einsum("...MI, HIO -> ...MHO", value, self.value_kernel)

		# Scale dot-product, doing the division to either query or key
		# instead of their product saves some computation
		depth = tf.constant(self.head_size, dtype=query.dtype)
		query /= tf.sqrt(depth)

		# Calculate dot product attention
		if edge is None:
			logits = tf.einsum("...NHO, ...MHO -> ...HNM", query, key)
		else:
			logits = tf.einsum("...NHO, ...NMHO -> ...HNM", query, key)
		attn_coef = self.softmax(logits, mask)

		# attention dropout
		attn_coef_dropout = self.dropout(attn_coef, training=training)

		# attention * value
		multihead_output = tf.einsum("...HNM, ...MHI-> ...NHI", attn_coef_dropout, value)

		# Run the outputs through another linear projection layer. Recombining heads
		# is automatically done.
		output = tf.einsum(
			"...NHI, HIO -> ...NO", multihead_output, self.projection_kernel
		)
		if self.projection_bias is not None:
			output += self.projection_bias

		if return_attn_coef:
			return output, attn_coef
		else:
			return output

	def get_config(self):
		config = super().get_config()
		config.update(
			head_size=self.head_size,
			num_heads=self.num_heads,
			query_size=self.query_size,
			key_size=self.key_size,
			value_size=self.value_size,
			edge_size=self.edge_size,
			output_size=self.output_size,
			dropout=self._droput_rate,
			use_projection_bias=self.use_projection_bias,
			kernel_initializer=tf.keras.initializers.serialize(self.kernel_initializer),
			kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer),
			kernel_constraint=tf.keras.constraints.serialize(self.kernel_constraint),
			bias_initializer=tf.keras.initializers.serialize(self.bias_initializer),
			bias_regularizer=tf.keras.regularizers.serialize(self.bias_regularizer),
			bias_constraint=tf.keras.constraints.serialize(self.bias_constraint),
		)
		return config

class TransformerBlock(tf.keras.layers.Layer):

	def __init__(
		self,
		head_size: int,
		num_heads: int,
		feed_forward_size: int,
		use_edge: bool = False,
		# if using edge feature, try setting pre_edge_size + pre_key_size < 100, otherwise (B * L * L * (edge_size + key_size)) could be out of memory
		pre_query_size: int = None,
		pre_edge_size: int = None,
		pre_key_size: int = None,
		pre_value_size: int = None,

		dropout: float = 0.0,
		use_projection_bias: bool = True,
		kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
		kernel_regularizer: typing.Union[str, typing.Callable] = None,
		kernel_constraint: typing.Union[str, typing.Callable] = None,
		bias_initializer: typing.Union[str, typing.Callable] = "zeros",
		bias_regularizer: typing.Union[str, typing.Callable] = None,
		bias_constraint: typing.Union[str, typing.Callable] = None,
		**kwargs,
	):

		super().__init__(**kwargs)
		self.use_edge = use_edge
		self.output_size = None
		if use_edge:
			self.pre_edge_size = pre_edge_size
		else:
			self.pre_edge_size = None
		self.pre_query_size = pre_query_size
		self.pre_key_size = pre_key_size
		self.pre_value_size = pre_value_size
		self.feed_forward_size = feed_forward_size
		self.kernel_settings = {}
		self.kernel_settings.update(
			kernel_initializer = kernel_initializer,
			kernel_regularizer = kernel_regularizer,
			kernel_constraint = kernel_constraint,
			bias_initializer = bias_initializer,
			bias_regularizer = bias_regularizer,
			bias_constraint = bias_constraint,
		)

		self.MHA = MultiHeadAttentionEdge(
			head_size = head_size,
			num_heads = num_heads,
			dropout = dropout,
			use_projection_bias = use_projection_bias,
			**self.kernel_settings
		)

		if self.pre_edge_size is not None:
			self.pre_edge = tf.keras.layers.Dense(
				self.edge_size, 
				**self.kernel_settings
			)

		if self.pre_query_size is not None:
			self.pre_query = tf.keras.layers.Dense(
				self.query_size, 
				**self.kernel_settings
			)
		
		if self.pre_key_size is not None:
			self.pre_key = tf.keras.layers.Dense(
				self.key_size, 
				**self.kernel_settings
			)

		if self.pre_value_size is not None:
			self.pre_value = tf.keras.layers.Dense(
				self.value_size, 
				**self.kernel_settings
			)
		
		self.FFNin = tf.keras.layers.Dense(
			self.feed_forward_size,
			activation = "relu",
			**self.kernel_settings
		)


		self.layernormMHA = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropoutFFN = tf.keras.layers.Dropout(dropout)
		self.layernormFFN = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def build(self, input_shape): # (node, edge) or node
		if isinstance(input_shape, (list, tuple)):
			node_size = input_shape[0][-1]
			if len(input_shape)==2:
				edge_size = input_shape[1][-1]
			else:
				edge_size = 0
		else:
			node_size = input_shape[-1]
			edge_size = 0
		if self.use_edge:
			if edge_size == 0:
				raise ValueError("must provide edge feature when layer.use_edge is set as true")
		else:
			edge_size = 0

		if self.pre_value_size is None:
			value_size = node_size
		else:
			value_size = self.pre_value_size

		if self.pre_query_size is None:
			query_size = node_size
		else:
			query_size = self.pre_query_size

		if self.pre_key_size is None:
			key_size = node_size
		else:
			key_size = self.pre_key_size

		self.FFNout = tf.keras.layers.Dense(
			value_size,
			**self.kernel_settings
		)
		self.output_size = value_size
		
		self.MHA.build(
			input_shape = (
				[None, query_size], # query
				[None, key_size], # key
				[None, value_size], # value
				[None, None, edge_size] # edge
			)
		)

	def call(self, inputs, return_attn_coef = False, mask = None, training = False): # (node, edge) or node
		if isinstance(inputs, (list, tuple)):
			node = inputs[0]
			if len(inputs)==2:
				edge = inputs[1]
			else:
				edge = None
		else:
			node = inputs
			edge = None
		if self.use_edge:
			if edge is None:
				raise ValueError("must provide edge feature when layer.use_edge is set as true")
			elif self.pre_edge_size is not None:
				edge = self.pre_edge(edge)
		else:
			edge = None

		if self.pre_value_size is not None:
			value = self.pre_value(node)
		else:
			value = node
		if self.pre_key_size is not None:
			key = self.pre_key(node)
		else:
			key = node
		if self.pre_query_size is not None:
			query = self.pre_query(node)
		else:
			query = node

		out = self.MHA((query, key, value, edge), return_attn_coef = return_attn_coef, mask = mask, training = training)
		x = self.layernormMHA(value + out)
		out = self.FFNin(x)
		out = self.FFNout(out)
		x = self.layernormFFN(x + out)

		if return_attn_coef:
			return x, return_attn_coef

		else:
			return x



