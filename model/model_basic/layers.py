import typing
import warnings
import tensorflow as tf
import tensorflow_probability as tfp

class RigidFrom3Points(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
	def call(self, x1, x2, x3):
		# 3D coordinates: shape: (L, 3 (as x,y,z)), x1: N, x2: Ca, x3: C
		v1 = x3 - x2
		v2 = x1 - x2
		e1 = v1 / tf.math.reduce_euclidean_norm(v1, -1, keepdims = True) # (L, 3) / (L, 1) -> (L, 3)
		u2 = v2 - e1 * (tf.reduce_sum(e1 * v2, axis = -1, keepdims = True)) # (L, 3) - (L, 3) * (L, 1) -> (L, 3)
		e2 = u2 / tf.math.reduce_euclidean_norm(u2, -1, keepdims = True) # (L, 3)
		e3 = tf.linalg.cross(e1, e2) # (L, 3)
		R = tf.stack([e1, e2, e3], axis = -1)
		# R: (L, 3(original xyz coordinates), 3'(new local coordinates)), ...GF in later representation
		return R, x2

class Dense(tf.keras.layers.Layer):
	def __init__(self, units: int, activation = None,
		use_bias = True,
		kernel_initializer="glorot_uniform",
		bias_initializer="zeros",
		kernel_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		**kwargs,
	):
		super().__init__(activity_regularizer=activity_regularizer, **kwargs)
		self.units = units
		self.activation = tf.keras.activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
		self.bias_constraint = tf.keras.constraints.get(bias_constraint)
		
	def build(self, input_shape):
		self.kernel = self.add_weight(
			"kernel",
			shape=[input_shape[-1], self.units],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
		)
		if self.use_bias:
			self.bias = self.add_weight(
				"bias",
				shape = [self.units],
				initializer=self.bias_initializer,
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint,
			)
		else:
			self.bias = None

	def call(self, inputs):
		outputs = tf.einsum("...I,IO->...O", inputs, self.kernel)
		if self.use_bias:
			outputs = outputs + self.bias
		if self.activation is not None:
			outputs = self.activation(outputs)
		return outputs


# IPA, with scaler, point, w/wo pairwise represention
class IPA(tf.keras.layers.Layer): 
	def __init__(
		self,
		num_heads: int,
		scaler_head_size: int,
		scaler_value_size: int,
		point_head_size: int,
		point_value_size: int,
		use_pairwise: bool,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.num_heads = num_heads
		self.scaler_head_size = scaler_head_size
		self.scaler_value_size = scaler_value_size
		self.point_head_size = point_head_size
		self.point_value_size = point_value_size
		self.use_pairwise = use_pairwise
		if use_pairwise:
			self.num_attn_logits = 3.
		else:
			self.num_attn_logits = 2.

		# point weight for each head
		point_weight_init = tf.math.log(tf.math.exp(tf.ones((self.num_heads,))) - 1.)
		self.point_weight = tf.Variable(point_weight_init, dtype = tf.float32)
		# constant scale for attention
		self.scaler_depth_scale = tf.sqrt(tf.cast(self.scaler_head_size, tf.float32))
		self.point_depth_scale = tf.sqrt(tf.cast(self.point_head_size * 9 / 2, tf.float32))
		self.softmax = tf.keras.layers.Softmax()

	def build(self, input_shape): # input: node, R, t, pairwise
		self.node_size = input_shape[0][-1]
		if self.use_pairwise:
			self.pairwise_size = input_shape[3][-1]
		else:
			self.pairwise_size = 0

		# projection for scaler part
		self.scaler_query_kernel = self.add_weight(
			name="scaler_query_kernel",
			shape=[self.num_heads, self.node_size, self.scaler_head_size],
			initializer="glorot_uniform",
		)
		self.scaler_key_kernel = self.add_weight(
			name="scaler_key_kernel",
			shape=[self.num_heads, self.node_size, self.scaler_head_size],
			initializer="glorot_uniform",
		)
		self.scaler_value_kernel = self.add_weight(
			name="scaler_value_kernel",
			shape=[self.num_heads, self.node_size, self.scaler_value_size],
			initializer="glorot_uniform",
		)
		# projection for point part
		self.point_query_kernel = self.add_weight(
			name="point_query_kernel",
			shape=[self.num_heads, self.node_size, self.point_head_size, 3],
			initializer="glorot_uniform",
		)
		self.point_key_kernel = self.add_weight(
			name="point_key_kernel",
			shape=[self.num_heads, self.node_size, self.point_head_size, 3],
			initializer="glorot_uniform",
		)
		self.point_value_kernel = self.add_weight(
			name="point_value_kernel",
			shape=[self.num_heads, self.node_size, self.point_value_size, 3],
			initializer="glorot_uniform",
		)
		# projection for pairwise part
		if self.use_pairwise:
			self.pairwise_kernel = self.add_weight(
				name = "pairwise_kernel",
				shape = [self.num_heads, self.pairwise_size],
				initializer = "glorot_uniform",
			)

		# output projection
		self.output_kernel = self.add_weight(
			name="output_kernel",
			shape=[self.num_heads, self.point_value_size * 4 + self.scaler_value_size + self.pairwise_size, self.node_size],
			initializer="glorot_uniform",
		)
		self.output_bias = self.add_weight(
			name="output_bias",
			shape=[self.node_size],
			initializer="zeros",
		)

	def call(self, inputs, return_attn_coef = False, mask = None):
		if self.use_pairwise:
			node, R, t, pairwise = inputs
		else:
			node, R, t = inputs
		if mask is not None:
			mask = tf.expand_dims(mask, -3) # mask: BLL, expand head dimension to BHLL
		# scaler part
		scaler_query = tf.einsum("HIK, ...LI -> ...HLK", self.scaler_query_kernel, node)
		scaler_key = tf.einsum("HIK, ...LI -> ...HLK", self.scaler_key_kernel, node)
		scaler_value = tf.einsum("HIV, ...LI -> ...HLV", self.scaler_value_kernel, node)
		scaler_query /= self.scaler_depth_scale
		scaler_logits = tf.einsum("...HNK, ...HMK -> ...HNM", scaler_query, scaler_key)

		# point part: transformation in local coords
		point_query = tf.einsum("HIKC, ...LI -> ...HLKC", self.point_query_kernel, node)
		point_key = tf.einsum("HIKC, ...LI -> ...HLKC", self.point_key_kernel, node)
		point_value = tf.einsum("HIVC, ...LI -> ...HLVC", self.point_value_kernel, node)
		# point part: project to global coords
		t = tf.expand_dims(t, -2)
		t = tf.expand_dims(t, -4)
		point_query = tf.einsum("...HLKF, ...LGF -> ...HLKG", point_query, R) + t
		point_key = tf.einsum("...HLKF, ...LGF -> ...HLKG", point_key, R) + t
		point_value = tf.einsum("...HLVF, ...LGF -> ...HLVG", point_value, R) + t
		# squared distance attention: Q, K: HLKC -> HL1KC - H1LKC
		point_qk_diff = tf.expand_dims(point_query, -3) - tf.expand_dims(point_key, -4)
		point_dist = tf.reduce_sum(tf.math.square(point_qk_diff), axis = [-1, -2]) # HLL
		point_weight = tf.math.softplus(self.point_weight) # H
		point_weight = tf.reshape(point_weight, (-1, 1, 1))
		point_logits = -0.5 * (point_dist * point_weight / self.point_depth_scale) # HLL
		# pairwise attention
		if self.use_pairwise:
			pairwise_logits = tf.einsum("HC,...MNC->...HMN", self.pairwise_kernel, pairwise)

		# add all
		if self.use_pairwise:
			logits = scaler_logits + pairwise_logits + point_logits
		else:
			logits = scaler_logits + point_logits
		attn_coef = self.softmax(logits / tf.sqrt(self.num_attn_logits), mask)
		scaler_output = tf.einsum("...HNM, ...HMV -> ...HNV", attn_coef, scaler_value)
		point_output = tf.einsum("...HNM, ...HMVC -> ...HNVC", attn_coef, point_value)
		if self.use_pairwise:
			pairwise_output = tf.einsum("...HNM, ...NMC -> ...HNC", attn_coef, pairwise)

		# rotate back to local
		point_output = tf.einsum("...HLVG, ...LGF -> ...HLVF", point_output - t, R)
		point_output_norm = tf.sqrt(tf.reduce_sum(tf.square(point_output), axis=-1) + 1e-12) # HLV
		s = tf.shape(point_output)
		point_output = tf.reshape(point_output, tf.concat([s[:-2], [s[-1]*s[-2]]], -1)) # HL(3V)
		if self.use_pairwise:
			result_output = tf.concat([scaler_output, pairwise_output, point_output, point_output_norm], -1) # HL(Vs+Vpair+3Vp+Vp)
		else:
			result_output = tf.concat([scaler_output, point_output, point_output_norm], -1)
		result_output = tf.einsum("...HIO, ...HLI -> ...LO", self.output_kernel, result_output) + self.output_bias
		if return_attn_coef:
			return result_output, attn_coef
		else:
			return result_output

class FFN(tf.keras.layers.Layer):
	def __init__(
		self,
		ff_multi: int = 1,
		ff_num_layer: int = 1,
		activation = "relu",
		**kwargs,
	):
		super().__init__(**kwargs)
		self.activation = activation
		self.n_layer = ff_num_layer
		self.ff_multi = ff_multi
	def build(self, input_shape):
		input_dim = input_shape[-1]
		layers = []
		for i in range(self.n_layer):
			layers.append(tf.keras.layers.Dense(self.ff_multi * input_dim, activation = self.activation))
		layers.append(tf.keras.layers.Dense(input_dim))
		self.layers = layers
	def call(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

class IPABlock(tf.keras.layers.Layer):
	def __init__(
		self,
		num_heads: int,
		scaler_head_size: int,
		scaler_value_size: int,
		point_head_size: int,
		point_value_size: int,
		use_pairwise: bool,
		ff_multi: int = 1,
		ff_num_layer: int = 1,
		activation = "relu",
		post_IPA_dropout: float = 0.,
		post_FFN_dropout: float = 0.,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.IPA = IPA(num_heads, scaler_head_size, scaler_value_size, point_head_size, point_value_size, use_pairwise)
		self.FFN = FFN(ff_multi, ff_num_layer, activation)
		self.layernormIPA = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernormFFN = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.post_IPA_dropout = tf.keras.layers.Dropout(post_IPA_dropout)
		self.post_FFN_dropout = tf.keras.layers.Dropout(post_FFN_dropout)
	def build(self, input_shape): # node, R, t, pairwise
		self.IPA.build(input_shape)
		self.FFN.build(input_shape[0])
	def call(self, inputs, mask = None, training = False, return_attn_coef = False): # node, R, t
		if return_attn_coef:
			x, attn_coef = self.IPA(inputs, mask = mask, return_attn_coef = True)
		else:
			x = self.IPA(inputs, mask = mask, return_attn_coef = False)
		x = self.post_IPA_dropout(x + inputs[0], training = training)
		x = self.layernormIPA(x)
		y = self.FFN(x)
		y = self.post_FFN_dropout(y + x, training = training)
		y = self.layernormFFN(y)
		if return_attn_coef:
			return y, attn_coef
		else:
			return y

class Evol(tf.keras.layers.Layer):
	def __init__(
		self,
		weighting = "spe",
		**kwargs,
	):
		super().__init__(**kwargs)
		self.weighting = weighting

	def build(self, input_shape): # input_shape: (L, N, A)
		self.aa_size = input_shape[-1]
		self.num_species = input_shape[-2]
		self.length = input_shape[-3]
		if self.weighting == "spe":
			self.W = self.add_weight(
				name = "species_weights",
				shape = (self.num_species),
				initializer = tf.keras.initializers.Constant(0.)
			)
		else:
			self.W = tf.constant(
				1. / self.num_species,
				shape = (self.num_species),
				dtype = tf.float32
			)

	def call(self, x):
		if self.weighting == "spe":
			W = tf.nn.softmax(self.W, axis=-1)
		else:
			W = self.W
		x1 = tf.einsum("...N, ...LNA -> ...LA", W, x)
		x2 = x - tf.expand_dims(x1, -2) # (L, N, A)
		x2 = tf.sqrt(tf.expand_dims(W, -1)) * x2 # (L, N, A)
		x3 = tf.einsum("...LNA, ...lNa -> ...LlAa", x2, x2)
		s = tf.shape(x3)
		x3 = tf.reshape(x3, tf.concat([s[:-2], [s[-1]*s[-2]]], -1)) # (L, L, A * A)
		norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=-1) + 1e-12)
		x4 = tf.concat([x3, tf.expand_dims(norm, -1)], axis = -1)
		return x4

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

		# Linear transformations
		if edge is None:
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
		if mask is not None:
			mask = tf.expand_dims(mask, -3) # expand head dimension
		attn_coef = self.softmax(logits, mask)

		# attention dropout
		attn_coef_dropout = self.dropout(attn_coef, training=training)

		# attention * value
		multihead_output = tf.einsum("...HNM, ...MHO-> ...NHO", attn_coef_dropout, value)

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

class TransformerBlock(tf.keras.layers.Layer):
	def __init__(
		self,
		head_size: int,
		num_heads: int,
		use_edge: bool = False,
		# if using edge feature, try setting pre_edge_size + pre_key_size < 100, otherwise (B * L * L * (edge_size + key_size)) could be out of memory
		pre_query_size: int = None,
		pre_edge_size: int = None,
		pre_key_size: int = None,
		pre_value_size: int = None,
		pre_activation: str = None,
		activation: str = "relu",
		ff_multi: int = 1,
		ff_num_layer: int = 1,
		
		post_MHA_dropout: float = 0.0,
		post_FFN_dropout: float = 0.0,
		
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
		self.pre_activation = pre_activation
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
			use_projection_bias = use_projection_bias,
			**self.kernel_settings
		)

		if self.pre_edge_size is not None:
			self.pre_edge = tf.keras.layers.Dense(
				self.pre_edge_size, 
				activation = self.pre_activation,
				**self.kernel_settings
			)

		if self.pre_query_size is not None:
			self.pre_query = tf.keras.layers.Dense(
				self.pre_query_size, 
				activation = self.pre_activation,
				**self.kernel_settings
			)
		
		if self.pre_key_size is not None:
			self.pre_key = tf.keras.layers.Dense(
				self.pre_key_size, 
				activation = self.pre_activation,
				**self.kernel_settings
			)

		if self.pre_value_size is not None:
			self.pre_value = tf.keras.layers.Dense(
				self.pre_value_size,
 				activation = self.pre_activation,
				**self.kernel_settings
			)
		
		self.FFN = FFN(ff_multi = ff_multi, ff_num_layer = ff_num_layer, activation = activation)

		self.post_MHA_dropout = tf.keras.layers.Dropout(post_MHA_dropout)
		self.post_FFN_dropout = tf.keras.layers.Dropout(post_FFN_dropout)
		self.layernormMHA = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernormFFN = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def build(self, input_shape): # (node, edge) or node
		if isinstance(input_shape, (list, tuple)):
			node_size = input_shape[0][-1]
			if len(input_shape)==2:
				input_edge_size = input_shape[1][-1]
			else:
				input_edge_size = 0
		else:
			node_size = input_shape[-1]
			input_edge_size = 0
		if self.use_edge:
			if input_edge_size == 0:
				raise ValueError("must provide edge feature when layer.use_edge is set as true")
		else:
			input_edge_size = 0

		if self.pre_edge_size is None:
			edge_size = input_edge_size
		else:
			edge_size = self.pre_edge_size

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

		if return_attn_coef:
			out, attn_coef = self.MHA((query, key, value, edge), return_attn_coef = return_attn_coef, mask = mask, training = training)
		else:
			out = self.MHA((query, key, value, edge), return_attn_coef = return_attn_coef, mask = mask, training = training)
		x = self.post_MHA_dropout(value + out, training = training)
		x = self.layernormMHA(x)
		
		out = self.FFN(x)
		x = self.post_FFN_dropout(x + out, training = training)
		x = self.layernormFFN(x)

		if return_attn_coef:
			return x, attn_coef

		else:
			return x

class AddGaussianNoise(tf.keras.layers.Layer):
	def __init__(self, last_dim = None, noise = 1., const = 4., **kwargs):
		super().__init__(**kwargs)
		self.noise = noise
		self.const = const
		self.last_dim = last_dim # only used to scale noise, do affect model structure
	def build(self, input_shape):
		if self.last_dim is None:
			self.last_dim = input_shape[-1]
	def call(self, data, mask = None, training = False):
		if not training:
			return data
		if mask is None:
			mask = tf.ones(tf.shape(data)[:-1])
		std = tf.math.reduce_std(tf.boolean_mask(data, mask), axis = 0)
		sample = tfp.distributions.Normal(loc = 0., scale = std).sample(tf.shape(data)[:-1])
		return data + sample * self.noise * tf.math.pow(tf.reduce_sum(mask), -1. / (self.const + self.last_dim))

class SymAttention(tf.keras.layers.Layer):
	def __init__(self, 
	             num_heads: int = 2,
	             head_size: int = 8,
	             kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
	             kernel_regularizer: typing.Union[str, typing.Callable] = None,
	             kernel_constraint: typing.Union[str, typing.Callable] = None,
	             **kwargs):
		super().__init__(**kwargs)
		self.num_heads = num_heads
		self.head_size = head_size
		self.kernel_initializer = kernel_initializer
		self.kernel_regularizer = kernel_regularizer
		self.kernel_constraint = kernel_constraint
	def build(self, input_shape):
		self.kernel = self.add_weight(
			name="attention_kernel",
			shape=[self.num_heads, self.num_heads, input_shape[-1], self.head_size],
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint,
		)
	def call(self, inputs): # inputs: (B, L, D)
		q = tf.einsum("...LD,KkDd->...KkLd", inputs, self.kernel)
		q /= tf.sqrt(tf.cast(head_size, tf.float32))
		k = tf.einsum("...LD,KkDd->...kKLd", inputs, self.kernel)
		J = tf.einsum("...KkLd,...Kkld->...LlKk", q, k)
		return J
		
class FullPotts(tf.keras.layers.Layer):
	def __init__(self, ref_species_index = 0, similarity_k = 10., similarity_mid = 0.8, **kwargs):
		super().__init__(**kwargs)
		self.ref_species_index = ref_species_index
		self.similarity_k = similarity_k
		self.similarity_mid = similarity_mid
	def call(self, inputs):
		MSA, h, J, length_weight, depth_weight = inputs
		MSA = tf.einsum('...LMA->...MLA', MSA) # transpose
		# MSA: (B, M, L, A+1)
		# h: (B, L, A+1) node
		# J: (B, L, L, A+1, A+1) edge interaction
		# length_weight: (B, L, 1)
		# depth_weight: (B, M)
		# calculating H difference cased by h for each sequence alternating each AA
		h = tf.expand_dims(h, -3) # (B, 1, L, A+1)
		J = tf.expand_dims(J, 1)
		H_node_wt = tf.reduce_sum(MSA * h, axis = -1, keepdims = True) # (B, M, L, 1)
		H_node_diff = h - H_node_wt # (B, M, L, A+1)
		# calculating H difference cased by J for pairwise AA
		mut = tf.ones((tf.shape(MSA)[-2]))
		ref = tf.einsum("...MLA,l->...MlLA", MSA, mut) # (B, M, L, L, A+1)
		ref = tf.expand_dims(ref, -2) # (B, M, L, L, 1, A+1)
		mut_interact = tf.reduce_sum(ref * J, -1) # (B, M, L, L, A+1)
		H_edge_mut = tf.reduce_sum(mut_interact, -2) # (B, M, L, A+1)
		H_edge_wt = tf.reduce_sum(H_edge_mut * MSA, axis = -1, keepdims = True)
		H_edge_diff = H_edge_mut - H_edge_wt
		H_diff = H_node_diff + H_edge_diff
		H_diff = H_diff[:,:,:,:-1] # (B, M, L, A)
		logits = - H_diff - tf.math.reduce_logsumexp(-H_diff, axis = -1, keepdims = True) # (B, M, L, A)
		logp_msa = tf.reduce_sum(logits * MSA[:,:,:,:-1], axis = -1) # (B, M, L)
		# weighted by length and effective depth
		same_aa = tf.einsum('...MLA,...NLA->...MNL', MSA, MSA) # (B, M, M, L)
		same_aa = tf.einsum('...MNL,...Ld->...MNd', same_aa, length_weight) # (B, M, M, 1)
		similarity_matrix = tf.reduce_sum(same_aa, axis = -1) / tf.reduce_sum(length_weight, axis = -2, keepdims = True) # (B, M, M)
		similarity_weight = 1 / tf.reduce_sum(tf.math.sigmoid(self.similarity_k * (similarity_matrix - self.similarity_mid)), axis = -1) # (B, M)
		similarity_weight *= depth_weight
		logp_msa = tf.einsum('...ML,...Ld->...Md', logp_msa, length_weight)
		logp_msa = tf.reduce_sum(logp_msa, axis = -1) # (B, M)
		logp_msa = tf.reduce_sum(similarity_weight * logp_msa, axis = -1) / tf.reduce_sum(similarity_weight, axis = -1) # (B)
		H_diff = tf.gather(H_diff, indices = self.ref_species_index, axis = -3)
		return H_diff, logp_msa # (B, L, A), (B)

