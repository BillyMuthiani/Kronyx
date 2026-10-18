# Examples

## XOR Problem

Binary classification of XOR inputs.

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, 8))
model.add(ReLU())
model.add(Dense(8, 1))
model.add(Sigmoid())
model.compile(loss=BinaryCrossEntropy(), optimizer=Adam(), metric=Accuracy())
model.fit(X, y, epochs=1000)
```

## Iris Classification

Multi-class classification.

```bash
python examples/iris_classification.py
```

## MNIST Digit Classification

The `mnist_classifier.py` example uses scikit-learn's `load_digits` dataset,
which provides 8×8 grayscale digit images. Run it with:

```bash
python examples/mnist_classifier.py
```

> **Note:** This example uses the sklearn digits dataset (1797 samples, 8×8 images),
> not the original MNIST database (28×28 images). Both are handwritten digit
> classification tasks, but the input dimensions differ.

## Serialization

Save/load model workflow.

```python
model.save('model.krx')
loaded = load_model('model.krx')
```