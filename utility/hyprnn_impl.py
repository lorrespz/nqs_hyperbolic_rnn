#Code written by HLD by adapting the following TF1 code
#https://github.com/dalab/hyperbolic_nn
import hyp_util as util
import tensorflow as tf
import keras
from tensorflow.keras import ops

#https://www.tensorflow.org/api_docs/python/tf/keras/Layer
#https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling

class EuclRNN(tf.keras.Layer):
    def __init__(self, num_units,  dtype):
        super().__init__()
        self._num_units = num_units
        #self.eucl_vars = []

    #argument of 'build' is fixed - do not rename !
    def build(self, input_shape):
        input_depth = input_shape[-1]
        self.W = self.add_weight(shape=[self._num_units, self._num_units],
                                initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", 
                                                                                distribution="uniform"),
                                trainable=True,
                                name="W")

        self.U = self.add_weight(shape=[input_depth, self._num_units],
                                initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", 
                                                                                distribution="uniform"),
                                trainable=True,
                                name="U")

        self.b = self.add_weight(shape=[1, self._num_units],
                                initializer=tf.constant_initializer(value=0, support_partition=False),
                                trainable=True,
                                name="b")

    def call(self, inputs, state):
        new_h = tf.math.tanh(ops.matmul(state, self.W) + ops.matmul(inputs, self.U) + self.b)
        return new_h, new_h
################################################################################################
class EuclGRU(tf.keras.Layer):
    def __init__(self, num_units, dtype):
        super().__init__()
        self._num_units = num_units
        #self.eucl_vars = []

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        input_depth = input_shape[-1]
        self.Wz = self.add_weight(shape=[self._num_units, self._num_units],
                                 initializer=tf.keras.initializers.VarianceScaling(scale=1.0, 
                                                                                            mode="fan_avg", 
                                                                                            distribution="uniform"),
                                trainable=True,
                                name="Wz")

        self.Uz = self.add_weight(shape=[input_depth, self._num_units],
                                 initializer=tf.keras.initializers.VarianceScaling(scale=1.0, 
                                                                                            mode="fan_avg", 
                                                                                            distribution="uniform"),
                                trainable=True,
                                name="Uz")

        self.bz = self.add_weight(shape=[1, self._num_units],
                                 initializer=tf.constant_initializer(value = 0),
                                 trainable = True,
                                 name = 'bz')

        self.Wr = self.add_weight(shape=[self._num_units, self._num_units],
                                initializer=tf.keras.initializers.VarianceScaling(scale=1.0, 
                                                                                mode="fan_avg", 
                                                                                distribution="uniform"),
                                trainable=True,
                                name = 'Wr')

        self.Ur = self.add_weight(shape=[input_depth, self._num_units],
                                initializer=tf.keras.initializers.VarianceScaling(scale=1.0, 
                                                                                mode="fan_avg", 
                                                                                distribution="uniform"),
                                trainable=True,
                                name = 'Ur')

        self.br = self.add_weight(shape=[1, self._num_units],
                                 initializer=tf.constant_initializer(value = 0),
                                 trainable = True,
                                 name = 'br')

        self.Wh = self.add_weight(shape=[self._num_units, self._num_units],
                                initializer=tf.keras.initializers.VarianceScaling(scale=1.0, 
                                                                                mode="fan_avg", 
                                                                                distribution="uniform"),
                                trainable=True,
                                name = 'Wh')

        self.Uh = self.add_weight(shape=[input_depth, self._num_units],
                                initializer=tf.keras.initializers.VarianceScaling(scale=1.0, 
                                                                                mode="fan_avg", 
                                                                                distribution="uniform"),
                                trainable=True,
                                name = 'Uh')

        self.bh = self.add_weight(shape=[1, self._num_units],
                                 initializer=tf.constant_initializer(value = 0),
                                 trainable = True,
                                 name = 'bh')

    def call(self, inputs, state):    
        z = tf.math.sigmoid(ops.matmul(state, self.Wz) + ops.matmul(inputs, self.Uz) + self.bz)
        r = tf.math.sigmoid(ops.matmul(state, self.Wr) + ops.matmul(inputs, self.Ur) + self.br)
        h_tilde = tf.math.tanh(ops.matmul(r * state, self.Wh) + ops.matmul(inputs, self.Uh) + self.bh)

        new_h = (1 - z) * state + z * h_tilde
        return new_h, new_h

################################################################################################
class HypRNN(tf.keras.Layer):

    def __init__(self,
                 num_units,
                 inputs_geom,
                 bias_geom,
                 c_val,
                 non_lin,
                 dtype):
        super().__init__()
        self._num_units = num_units
        self.c_val = c_val
        self.__dtype = dtype
        self.non_lin = non_lin
        assert self.non_lin in ['id', 'relu', 'tanh', 'sigmoid']

        self.bias_geom = bias_geom
        self.inputs_geom = inputs_geom
        assert self.inputs_geom in ['eucl', 'hyp']
        assert self.bias_geom in ['eucl', 'hyp']
  
        self.matrix_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", 
                                                                            distribution="uniform")
        self.eucl_vars = []
        self.hyp_vars = []

    # Performs the hyperbolic version of the operation Wh + Ux + b.
    def one_rnn_transform(self, W, h, U, x, b):
        hyp_x = x
        if self.inputs_geom == 'eucl':
            hyp_x = util.tf_exp_map_zero(x, self.c_val)

        hyp_b = b
        if self.bias_geom == 'eucl':
            hyp_b = util.tf_exp_map_zero(b, self.c_val)

        W_otimes_h = util.tf_mob_mat_mul(W, h, self.c_val)
        U_otimes_x = util.tf_mob_mat_mul(U, hyp_x, self.c_val)
        Wh_plus_Ux = util.tf_mob_add(W_otimes_h, U_otimes_x, self.c_val)
        result = util.tf_mob_add(Wh_plus_Ux, hyp_b, self.c_val)
        return result

    def build(self, input_shape):
        input_depth = input_shape[-1]

        self.W = self.add_weight(shape=[self._num_units, self._num_units],            
                                initializer=self.matrix_initializer,
                                trainable=True,
                                name = 'W')
        self.eucl_vars.append(self.W)

        self.U = self.add_weight(shape=[input_depth, self._num_units],            
                                initializer=self.matrix_initializer,
                                trainable=True,
                                name = 'Uh')
       
        self.eucl_vars.append(self.U)

        self.b = self.add_weight(shape=[1, self._num_units],
                                trainable=True,
                                initializer=tf.constant_initializer(value = 0.0),
                                name = 'b')

        if self.bias_geom == 'hyp':
            self.hyp_vars.append(self.b)
        else:
            self.eucl_vars.append(self.b)

    def call(self, inputs, state): 
        hyp_x = inputs
        if self.inputs_geom == 'eucl':
            hyp_x = util.tf_exp_map_zero(inputs, self.c_val)

        new_h = self.one_rnn_transform(self.W, state, self.U, hyp_x, self.b)
        new_h = util.tf_hyp_non_lin(new_h, non_lin=self.non_lin, hyp_output=True, c=self.c_val)
        return new_h, new_h
###############################################################################################
class HypGRU(tf.keras.Layer):

    def __init__(self,
                 num_units,
                 inputs_geom,
                 bias_geom,
                 c_val,
                 non_lin,
                 dtype):
        super().__init__()
        self._num_units = num_units
        self.c_val = c_val
        self.non_lin = non_lin
        assert self.non_lin in ['id', 'relu', 'tanh', 'sigmoid']
        self.bias_geom = bias_geom
        self.inputs_geom = inputs_geom
        assert self.inputs_geom in ['eucl', 'hyp']
        assert self.bias_geom in ['eucl', 'hyp']
   
        self.matrix_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

        self.eucl_vars = []
        self.hyp_vars = []

    # Performs the hyperbolic version of the operation Wh + Ux + b.
    def one_rnn_transform(self, W, h, U, x, b):
        hyp_b = b
        if self.bias_geom == 'eucl':
            hyp_b = util.tf_exp_map_zero(b, self.c_val)

        W_otimes_h = util.tf_mob_mat_mul(W, h, self.c_val)
        U_otimes_x = util.tf_mob_mat_mul(U, x, self.c_val)
        Wh_plus_Ux = util.tf_mob_add(W_otimes_h, U_otimes_x, self.c_val)
        return util.tf_mob_add(Wh_plus_Ux, hyp_b, self.c_val)

    def build(self, input_shape):
        input_depth = input_shape[-1]

        self.Wz = self.add_weight(shape=[self._num_units, self._num_units],
                                 trainable=True,
                                 initializer=self.matrix_initializer,
                                 name = 'W_z')
        self.eucl_vars.append(self.Wz)

        self.Uz = self.add_weight(shape=[input_depth, self._num_units],
                                trainable=True,
                                initializer=self.matrix_initializer,
                                name = 'U_z')
        self.eucl_vars.append(self.Uz)
                
        self.bz = self.add_weight(shape=[1, self._num_units],
                                trainable=True,
                                initializer=tf.compat.v1.constant_initializer(0.0), 
                                 name = 'b_z')
 
        if self.bias_geom == 'hyp':
            self.hyp_vars.append(self.bz)
        else:
            self.eucl_vars.append(self.bz)

        ###########################################
        self.Wr = self.add_weight(shape=[self._num_units, self._num_units],
                                trainable=True,
                                initializer=self.matrix_initializer,
                                name = 'W_r')
        self.eucl_vars.append(self.Wr)
                
        self.Ur = self.add_weight(shape=[input_depth, self._num_units],
                                trainable=True,
                                initializer=self.matrix_initializer,
                                name = 'U_r')
        self.eucl_vars.append(self.Ur)
                
        self.br = self.add_weight(shape=[1, self._num_units],
                                trainable=True,
                                initializer=tf.compat.v1.constant_initializer(0.0),
                                name = 'b_r')

        if self.bias_geom == 'hyp':
            self.hyp_vars.append(self.br)
        else:
            self.eucl_vars.append(self.br)
        ###########################################
        self.Wh = self.add_weight(shape=[self._num_units, self._num_units],
                                trainable=True,
                                initializer=self.matrix_initializer,
                                name = 'W_h')
        self.eucl_vars.append(self.Wh)
                
        self.Uh = self.add_weight(shape=[input_depth, self._num_units],
                                trainable=True,
                                initializer=self.matrix_initializer,
                                name = 'U_h')
        self.eucl_vars.append(self.Uh)
             
        self.bh = self.add_weight(shape=[1, self._num_units],
                                trainable=True,
                                initializer=tf.compat.v1.constant_initializer(0.0),
                                name = 'b_h')
        if self.bias_geom == 'hyp':
            self.hyp_vars.append(self.bh)
        else:
            self.eucl_vars.append(self.bh)        

    def call(self, inputs, state):
        hyp_x = inputs
        if self.inputs_geom == 'eucl':
            hyp_x = util.tf_exp_map_zero(inputs, self.c_val)

        z = util.tf_hyp_non_lin(self.one_rnn_transform(self.Wz, state, self.Uz, hyp_x, self.bz),
                                    non_lin='sigmoid',
                                    hyp_output=False,
                                    c = self.c_val)

        r = util.tf_hyp_non_lin(self.one_rnn_transform(self.Wr, state, self.Ur, hyp_x, self.br),
                                    non_lin='sigmoid',
                                    hyp_output=False,
                                    c = self.c_val)

        r_point_h = util.tf_mob_pointwise_prod(state, r, self.c_val)
        h_tilde = util.tf_hyp_non_lin(self.one_rnn_transform(self.Wh, r_point_h, self.Uh, hyp_x, self.bh),
                                          non_lin=self.non_lin,
                                          hyp_output=True,
                                          c=self.c_val)

        minus_h_oplus_htilde = util.tf_mob_add(-state, h_tilde, self.c_val)
        new_h = util.tf_mob_add(state,
                                    util.tf_mob_pointwise_prod(minus_h_oplus_htilde, z, self.c_val),
                                    self.c_val)
        return new_h, new_h
################################################################################################
