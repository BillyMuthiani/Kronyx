import numpy as np

from Neuralnet.model import Sequential
from Neuralnet.layers import Dense, Flatten
from Neuralnet.activations import ReLU, Softmax
from Neuralnet.losses import SoftmaxCategoricalCrossEntropy
from Neuralnet.optimizers import Adam
from Neuralnet.metrics import Accuracy


# Generate synthetic image-like data for demonstration
# 100 samples, 2x2 "images" (4 pixels), 3 classes
np.random.seed(42)

X_train = np.random.randn(100, 2, 2)  # batch=100, height=2, width=2
y_train = np.random.randint(0, 3, 100)

X_test = np.random.randn(20, 2, 2)
y_test = np.random.randint(0, 3, 20)

print(f'Input shape: {X_train.shape}')  # (100, 2, 2)


# Build Model with Flatten layer
model = Sequential()

# Flatten expects (batch, height, width) -> (batch, features)
model.add(Flatten())  # Output shape: (batch, 4)
model.add(Dense(4, 8))
model.add(ReLU())
model.add(Dense(8, 3))
model.add(Softmax())


# Train
model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)

model.fit(
    X_train,
    y_train,
    epochs=100
)


# Evaluate
predictions = model.predict(X_test)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

accuracy = np.mean(
    predicted_classes == y_test
)

print(
    f"Test Accuracy: {accuracy:.4f}"
)

# Demonstrate inference doesn't cache
print(f'Flatten shape cached: {model.layers[0].input_shape}')