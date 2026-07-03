import numpy as np


class SGD:

    def __init__(
        self,
        learning_rate=0.01
    ):

        self.learning_rate = learning_rate

    def update(self, layer):

        if hasattr(layer, "weights"):

            layer.weights -= (
                self.learning_rate
                * layer.dweights
            )

            layer.biases -= (
                self.learning_rate
                * layer.dbiases
            )

        if hasattr(layer, "gamma"):

            layer.gamma -= (
                self.learning_rate
                * layer.dgamma
            )

            layer.beta -= (
                self.learning_rate
                * layer.dbeta
            )


class Adam:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def update(self, layer):

        if hasattr(layer, "weights"):

            if not hasattr(layer, "m_w"):

                layer.m_w = np.zeros_like(
                    layer.weights
                )

                layer.v_w = np.zeros_like(
                    layer.weights
                )

                layer.m_b = np.zeros_like(
                    layer.biases
                )

                layer.v_b = np.zeros_like(
                    layer.biases
                )

            self.iterations += 1

            layer.m_w = (
                self.beta1 * layer.m_w
                +
                (1 - self.beta1)
                * layer.dweights
            )

            layer.m_b = (
                self.beta1 * layer.m_b
                +
                (1 - self.beta1)
                * layer.dbiases
            )

            layer.v_w = (
                self.beta2 * layer.v_w
                +
                (1 - self.beta2)
                * (layer.dweights ** 2)
            )

            layer.v_b = (
                self.beta2 * layer.v_b
                +
                (1 - self.beta2)
                * (layer.dbiases ** 2)
            )

            m_w_hat = (
                layer.m_w
                /
                (1 - self.beta1 ** self.iterations)
            )

            m_b_hat = (
                layer.m_b
                /
                (1 - self.beta1 ** self.iterations)
            )

            v_w_hat = (
                layer.v_w
                /
                (1 - self.beta2 ** self.iterations)
            )

            v_b_hat = (
                layer.v_b
                /
                (1 - self.beta2 ** self.iterations)
            )

            layer.weights -= (
                self.learning_rate
                * m_w_hat
                /
                (np.sqrt(v_w_hat) + self.epsilon)
            )

            layer.biases -= (
                self.learning_rate
                * m_b_hat
                /
                (np.sqrt(v_b_hat) + self.epsilon)
            )

        if hasattr(layer, "gamma"):

            if not hasattr(layer, "m_gamma"):

                layer.m_gamma = np.zeros_like(
                    layer.gamma
                )

                layer.v_gamma = np.zeros_like(
                    layer.gamma
                )

                layer.m_beta = np.zeros_like(
                    layer.beta
                )

                layer.v_beta = np.zeros_like(
                    layer.beta
                )

            self.iterations += 1

            layer.m_gamma = (
                self.beta1 * layer.m_gamma
                +
                (1 - self.beta1)
                * layer.dgamma
            )

            layer.m_beta = (
                self.beta1 * layer.m_beta
                +
                (1 - self.beta1)
                * layer.dbeta
            )

            layer.v_gamma = (
                self.beta2 * layer.v_gamma
                +
                (1 - self.beta2)
                * (layer.dgamma ** 2)
            )

            layer.v_beta = (
                self.beta2 * layer.v_beta
                +
                (1 - self.beta2)
                * (layer.dbeta ** 2)
            )

            m_gamma_hat = (
                layer.m_gamma
                /
                (1 - self.beta1 ** self.iterations)
            )

            m_beta_hat = (
                layer.m_beta
                /
                (1 - self.beta1 ** self.iterations)
            )

            v_gamma_hat = (
                layer.v_gamma
                /
                (1 - self.beta2 ** self.iterations)
            )

            v_beta_hat = (
                layer.v_beta
                /
                (1 - self.beta2 ** self.iterations)
            )

            layer.gamma -= (
                self.learning_rate
                * m_gamma_hat
                /
                (np.sqrt(v_gamma_hat) + self.epsilon)
            )

            layer.beta -= (
                self.learning_rate
                * m_beta_hat
                /
                (np.sqrt(v_beta_hat) + self.epsilon)
            )
