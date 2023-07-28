# architecture code is essentially the one of BayesFlow (10.1109/TNNLS.2020.3042395)
# you can refer to their paper for further details

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class CouplingNet(tf.keras.Model):
    """
    Implements the coupling net (s and t in our terminology)
    meta is a dictionary with the architecture parameters
    
    meta["activation"] is the activation function of the HIDDEN layers
    meta["initializer"] is the initializer of ALL the layers
    meta["units"] is an array of length N_layers where each entry is the number of nodes per layer
    """
    def __init__(self, meta, n_out):

        super(CouplingNet, self).__init__()

        self.dense = tf.keras.models.Sequential(
            # Hidden layer structure
            [tf.keras.layers.Dense(units, 
                   activation=meta['activation'], 
                   kernel_initializer=meta['initializer'])
             for units in meta['units']] +
            # Output layer
            [tf.keras.layers.Dense(n_out, kernel_initializer=meta['initializer'])]
        )

    def call(self, target, condition):
        """
        Performs a forward pass to the concatenation of target (model parameter)
        and condition (summary observables)
        """

        # Handle 3D case for a set-flow
        if len(target.shape) == 3:
            # Extract information about N
            N = int(target.shape[1])
            condition = tf.stack([condition] * N, axis=1)
        inp = tf.concat((target, condition), axis=-1)
        out = self.dense(inp)
        return out
    
class ConditionalCouplingLayer(tf.keras.Model):
    """
    Implements the conditional coupling layer
    meta is a dictionary with the architecture parameters
    
    meta["alpha"] is the clamping parameter (optional)
    meta["n_params"] is the dimension of the model parameter theta
    meta["s_args"], meta["t_args"] are arrays of length N_layers where each entry is the number of nodes per layer
    meta["use_permutation"] indicates whether to perform permutation (optional, not useful for D<3)
    """
    def __init__(self, meta):

        super(ConditionalCouplingLayer, self).__init__()

        # Coupling net initialization
        self.alpha = meta['alpha']
        theta_dim = meta['n_params']
        self.n_out1 = theta_dim // 2
        self.n_out2 = theta_dim // 2 if theta_dim % 2 == 0 else theta_dim // 2 + 1
            
        self.s1 = CouplingNet(meta['s_args'], self.n_out1)
        self.t1 = CouplingNet(meta['t_args'], self.n_out1)
        self.s2 = CouplingNet(meta['s_args'], self.n_out2)
        self.t2 = CouplingNet(meta['t_args'], self.n_out2)

        # Optional permutation
        if meta['use_permutation']:
            self.permutation = Permutation(theta_dim)
        else:
            self.permutation = None

    def _forward(self, target, condition):
        """ 
        Usual s, t forward calculation and combination for INN
        target = model parameters
        condition = summary observables
        """

        # Split parameter vector
        u1, u2 = tf.split(target, [self.n_out1, self.n_out2], axis=-1)

        # Pre-compute network outputs for v1
        s1 = self.s1(u2, condition)
        # Clamp s1 if specified
        if self.alpha is not None:
            s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
        t1 = self.t1(u2, condition)
        v1 = u1 * tf.exp(s1) + t1

        # Pre-compute network outputs for v2
        s2 = self.s2(v1, condition)
        # Clamp s2 if specified
        if self.alpha is not None:
            s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
        t2 = self.t2(v1, condition)
        v2 = u2 * tf.exp(s2) + t2
        v = tf.concat((v1, v2), axis=-1)

        # Compute ldj, # log|J| = log(prod(diag(J))) -> according to inv architecture
        log_det_J = tf.reduce_sum(s1, axis=-1) + tf.reduce_sum(s2, axis=-1)
        return v, log_det_J 

    def _inverse(self, z, condition):
        """
        Usual s, t inverse calculation and combination for INN
        """

        v1, v2 = tf.split(z, [self.n_out1, self.n_out2], axis=-1)

        # Pre-Compute s2
        s2 = self.s2(v1, condition)
        # Clamp s2 if specified
        if self.alpha is not None:
            s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
        u2 = (v2 - self.t2(v1, condition)) * tf.exp(-s2)

        # Pre-Compute s1
        s1 = self.s1(u2, condition)
        # Clamp s1 if specified
        if self.alpha is not None:
            s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
        u1 = (v1 - self.t1(u2, condition)) * tf.exp(-s1)
        u = tf.concat((u1, u2), axis=-1)

        return u

    def call(self, target_or_z, condition, inverse=False):
        """
        Performs one pass through an invertible chain (either inverse or forward).

        target_or_z = model parameters or normal samples from latent distribution
        condition = summary observables
        inverse = whether it is a forward or inverse pass
        """
        
        if not inverse:
            return self.forward(target_or_z, condition)
        return self.inverse(target_or_z, condition)

    def forward(self, target, condition):
        """Performs a forward pass and returns the "normalized" variables and the log determinant of the Jacobian"""

        # Initialize log_det_Js accumulator
        log_det_Js = tf.zeros(1)

        # Permute, if indicated
        if self.permutation is not None:
            target = self.permutation(target)

        # Pass through coupling layer
        z, log_det_J_c = self._forward(target, condition)
        log_det_Js += log_det_J_c

        return z, log_det_Js

    def inverse(self, z, condition):
        """Performs an inverse pass and returns the transformed normal samples"""
        
        # Pass through coupling layer
        target = self._inverse(z, condition)

        # Pass through optional permutation
        if self.permutation is not None:
            target = self.permutation(target, inverse=True)
        
        return target

class Permutation(tf.keras.Model):
    """
    Implements a permutation layer to permute the input dimensions of the cINN block.
    """

    def __init__(self, input_dim):
        """
        input_dim  = Dimensionality of the input to the cINN block (theta dimension)
        """

        super(Permutation, self).__init__()

        permutation_vec = np.random.permutation(input_dim)
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(initial_value=permutation_vec,
                                       trainable=False,
                                       dtype=tf.int32,
                                       name='permutation')
        self.inv_permutation = tf.Variable(initial_value=inv_permutation_vec,
                                           trainable=False,
                                           dtype=tf.int32,
                                           name='inv_permutation')

    def call(self, target, inverse=False):
        """
        Performes the permutation (or inverse permutation)
        """

        if not inverse:
            return tf.transpose(tf.gather(tf.transpose(target), self.permutation))
        return tf.transpose(tf.gather(tf.transpose(target), self.inv_permutation))

class Summary(tf.keras.Model):

    """
    Implements the summary network
    meta is a dictionary with the architecture parameters
    
    meta["n_units_summary"] is an array of length N_layers where each entry is the number of nodes per layer
    meta["summary"] is the output dimension of the summary network (so the number of summary observables to extract from x)
    """
    def __init__(self, meta):


        super(Summary, self).__init__()

        self.sumnet = tf.keras.models.Sequential(
            # Hidden layer structure
            [tf.keras.layers.Dense(units, 
                   activation=tf.keras.layers.LeakyReLU(alpha = 0.01))
             for units in meta["n_units_summary"]] +
            # Output layer
            [tf.keras.layers.Dense(meta["summary"])]
        )

    def call(self, condition):

        out = self.sumnet(condition)
        return out
    
class RealNVP_sum(keras.Model):
    """
    Implements the cINN (by combining previous parts)
    meta is a dictionary with the architecture parameters
    
    meta["n_coupling_layers"] is the number of coupling layers (their architecture is specified when calling CouplingNet)
    meta["n_params"] is the dimension of the model parameter theta
    """
    def __init__(self, meta):
        super(RealNVP_sum, self).__init__()
        self.meta = meta
        
        

        self.num_coupling_layers = self.meta['n_coupling_layers']

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=([0.,]*self.meta['n_params'] ), scale_diag=([1.,]*self.meta['n_params'] )
        )
        self.coupling_layers = [ConditionalCouplingLayer(self.meta) for _ in range(self.num_coupling_layers)]
        
        self.summary_net = Summary(self.meta)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        # self.layers_list = [Coupling(input_dim ) for i in range(num_coupling_layers)]



    @property
    def metrics(self):
        """
        List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    
    
    # summary added in forward and inverse pass
    def forward(self, target, condition):
        """Performs a forward pass though the chain."""

        condition_sum = self.summary_net(condition)
        z = target
        log_det_Js = []
        for layer in self.coupling_layers:
            z, log_det_J = layer(z, condition_sum)
            log_det_Js.append(log_det_J)
        # Sum Jacobian determinants for all layers (coupling blocks) to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)

        return z, log_det_J

    def inverse(self, z, condition):
        """Performs a reverse pass through the chain."""
        
        condition_sum = self.summary_net(condition)

        target = z
        for layer in reversed(self.coupling_layers):
            target = layer(target, condition_sum, inverse=True)
        return target

    def call(self, x, training=True):

        target, condition= x
    

        if training:
            return self.forward(target, condition)
        else:
            return self.inverse(target, condition)
    # Log likelihood of the normal distribution plus the log determinant of the jacobian.



    def log_loss(self, data):
        y, logdet = self(data)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    def log_like(self, data):
        y, logdet = self(data)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return log_likelihood

    def train_step(self, data):
        with tf.GradientTape() as tape:

            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
  
