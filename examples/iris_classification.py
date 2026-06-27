import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from Neuralnet.model import Sequential
from Neuralnet.layers import Dense
from Neuralnet.activations import ReLU, Softmax
from Neuralnet.losses import SoftmaxCategoricalCrossEntropy
from Neuralnet.optimizers import Adam
from Neuralnet.metrics import Accuracy
# Optional: Uncomment to save best model during training
# from Neuralnet.callbacks import ModelCheckpoint


# Step 3: Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)


# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 5: Normalize Features
X_train = (
    X_train - X_train.mean(axis=0)
) / X_train.std(axis=0)

X_test = (
    X_test - X_test.mean(axis=0)
) / X_test.std(axis=0)


# Step 6: Build Model
model = Sequential()

model.add(Dense(4, 16))
model.add(ReLU())

model.add(Dense(16, 3))
model.add(Softmax())


# Step 7: Train
model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    metric=Accuracy()
)

model.fit(
    X_train,
    y_train,
    epochs=5000
    # Optional: Add callbacks=[ModelCheckpoint("checkpoints/best.npz")] for auto-save
)


# Step 8: Evaluate
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