import numpy as np


class ReLU:

    def forward(self, X):

        self.input = X

        return np.maximum(0, X)

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        self.dinputs[self.input <= 0] = 0

        return self.dinputs


class Sigmoid:

    def forward(self, X):

        self.output = 1 / (1 + np.exp(-X))

        return self.output

    def backward(self, dvalues):

        self.dinputs = (
            dvalues
            * self.output
            * (1 - self.output)
        )

        return self.dinputs


class Tanh:

    def forward(self, X):

        self.output = np.tanh(X)

        return self.output

    def backward(self, dvalues):

        self.dinputs = (
            dvalues
            * (1 - self.output ** 2)
        )

        return self.dinputs


class Softmax:

    def forward(self, X):

        exp_values = np.exp(
            X - np.max(X, axis=1, keepdims=True)
        )

        self.output = (
            exp_values
            /
            np.sum(
                exp_values,
                axis=1,
                keepdims=True
            )
        )

        return self.output

    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        for index, (
            single_output,
            single_dvalues
        ) in enumerate(
            zip(self.output, dvalues)
        ):

            single_output = single_output.reshape(
                -1,
                1
            )

            jacobian = (
                np.diagflat(single_output)
                -
                np.dot(
                    single_output,
                    single_output.T
                )
            )

            self.dinputs[index] = np.dot(
                jacobian,
                single_dvalues
            )

        return self.dinputs