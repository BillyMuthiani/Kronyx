"""BatchLoader example.

Learn how to use BatchLoader for efficient batch processing
during neural network training.
"""

from kronyx import set_seed
from kronyx.data import BatchLoader, TensorDataset, train_test_split
from kronyx.datasets import blobs

# Set seed for reproducibility
set_seed(42)

# Generate dataset
X, y = blobs(samples=100, centers=2)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create BatchLoader
loader = BatchLoader(
    X_train,
    y_train,
    batch_size=16,
    shuffle=True,
    seed=42
)

print(f"\nNumber of batches: {len(loader)}")

# Iterate over batches
print("\nIterating over batches:")
for i, (_X_batch, y_batch) in enumerate(loader):
    print(f"  Batch {i+1}: X shape = {_X_batch.shape}, y shape = {y_batch.shape}")
    if i >= 2:
        print("  ...")
        break

# Demonstrate drop_last
loader_drop = BatchLoader(
    X_train,
    y_train,
    batch_size=30,
    shuffle=False,
    drop_last=True
)
print(f"\nWith drop_last=True, batches: {len(loader_drop)}")

# TensorDataset example
dataset = TensorDataset(X_train, y_train)
print(f"\nTensorDataset length: {len(dataset)}")
X0, y0 = dataset[0]
print(f"First sample: X={X0}, y={y0}")
