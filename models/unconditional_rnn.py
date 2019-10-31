import tensorflow as tf


class UnconditionalRNN:
    def __init__(self, num_layers=3, num_cells=400, num_mixtures=20):
        self.input_size = 3
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.num_cells = num_cells
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.build_model()

    def build_model(self):
        lstm = tf.keras.layers.LSTM
        concat = tf.keras.layers.concatenate

        # input layer
        self.inputs = tf.keras.Input(shape=(None, self.input_size))
        x = self.inputs

        # build RNN layers with skip connections to input
        rnns = []
        lstm_states = []
        x = lstm(self.num_cells, input_shape=(None, self.input_size), return_sequences=True)(x)
        rnns.append(x)
        for i in range(self.num_layers - 1):
            output_rnn = concat([self.inputs, x])
            x = lstm(self.num_cells,
                     input_shape=(None, self.num_cells + self.input_size),
                     return_sequences=True)(output_rnn)
            rnns.append(x)

        # two-dimensional mean and standard deviation, scalar correlation, weights
        params_per_mixture = 6
        output_rnn = concat([rnn for rnn in rnns]) # output skip connections
        self.outputs = tf.keras.layers.Dense(params_per_mixture * self.num_mixtures + 1,
                                        input_shape=(self.num_layers * self.num_cells, ))(output_rnn)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        self.model.summary()

    # Equations (18 - 23)
    def output_vector(self, outputs):
        e_hat = outputs[:, 0]
        pi_hat, mu_hat1, mu_hat2, sigma_hat1, sigma_hat2, rho_hat = tf.split(outputs[:, 1:], 6, 1)
        
        # calculate actual values
        end_stroke = tf.math.sigmoid(e_hat)
        mixture_weight = tf.math.softmax(pi_hat, axis=1)
        mean1 = mu_hat1
        mean2 = mu_hat2
        stddev1 = tf.math.exp(sigma_hat1)
        stddev2 = tf.math.exp(sigma_hat2)
        correl = tf.math.tanh(rho_hat)
        return [end_stroke, mixture_weight, mean1, mean2, stddev1, stddev2, correl]

    # Equation (24, 25)
    def bivariate_gaussian(self, x1, x2, stddev1, stddev2, mean1, mean2, correl):
        Z = tf.math.square((x1 - mean1) / stddev1) + tf.math.square((x2 - mean2) / stddev2) \
            - (2 * correl * (x1 - mean1) * (x2 - mean2) / (stddev1 * stddev2))
        return tf.math.exp(-Z / (2 * (1 - tf.math.square(correl)))) \
            / (2 * np.pi * stddev1 * stddev2 * tf.math.sqrt(1 - tf.math.square(correl)))

    # Equation (26)
    def loss(self, x1, x2, x3, stddev1, stddev2, mean1, mean2, correl, mixture_weight, end_stroke):
        gaussian = self.bivariate_gaussian(x1, x2, stddev1, stddev2, mean1, mean2, correl)
        left_term_inner = tf.reduce_sum(tf.math.multiply(mixture_weight, gaussian), 1, keep_dims=True)
        left_term = -tf.math.log(left_term_inner)
        right_term = -tf.math.log(tf.cond(x3 == 1, lambda: end_stroke, lambda: 1 - end_stroke))
        return tf.reduce_sum(left_term + right_term)

    @tf.function
    def train_step(self, inputs):
        loss = 0
        with tf.GradientTape() as tape:
            pass
        batch_loss = (loss / int(inputs.shape[0]))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def generate_handwriting(self):
        pass
