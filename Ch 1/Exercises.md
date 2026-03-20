# PyTorch Workflow Exercises

These exercises are based on the concepts covered in your `C1.ipynb` notebook.  
Each solution is hidden inside a collapsible section so you can try the problem first.

---

## Exercise 1: Create Simple Linear Data
Create dummy data using the linear equation:

$y = wx + b$

with:
- `w = 0.7`
- `b = 0.3`
- `X` ranging from `0` to `1` with step size `0.02`

Then print both `X` and `y`.

<details>
<summary>Hidden solution</summary>

```python
import torch

weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02)
y = weight * X + bias

print(X)
print(y)
```

</details>

---

## Exercise 2: Train/Test Split
Using the data from Exercise 1, split the dataset so that:
- 80% goes to training
- 20% goes to testing

Then print the lengths of each split.

<details>
<summary>Hidden solution</summary>

```python
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))
```

</details>

---

## Exercise 3: Visualize the Data
Write a helper function using `matplotlib` to scatter plot:
- training data in one color
- test data in another color

<details>
<summary>Hidden solution</summary>

```python
import matplotlib.pyplot as plt

def plot_data(train_data, train_labels, test_data, test_labels):
    plt.figure(figsize=(8, 5))
    plt.scatter(train_data, train_labels, label="Train data")
    plt.scatter(test_data, test_labels, label="Test data")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()

plot_data(X_train, y_train, X_test, y_test)
```

</details>

---

## Exercise 4: Build a Linear Regression Model Class
Create a PyTorch model class called `LinearRegressionModel` that:
- subclasses `nn.Module`
- has a learnable weight parameter
- has a learnable bias parameter
- implements `forward()` as `weight * x + bias`

<details>
<summary>Hidden solution</summary>

```python
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        return self.weights * x + self.bias
```

</details>

---

## Exercise 5: Inspect Model Parameters
Instantiate your model with a manual seed and print:
- the model parameters using `list(model.parameters())`
- the model state dictionary using `model.state_dict()`

<details>
<summary>Hidden solution</summary>

```python
torch.manual_seed(42)

model_0 = LinearRegressionModel()

print(list(model_0.parameters()))
print(model_0.state_dict())
```

</details>

---

## Exercise 6: Set Up Loss Function and Optimizer
For this regression task:
- use Mean Absolute Error loss
- use stochastic gradient descent with learning rate `0.01`

<details>
<summary>Hidden solution</summary>

```python
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)
```

</details>

---

## Exercise 7: One Training Step by Hand
Write code for a single training step:
1. set model to train mode
2. do a forward pass on `X_train`
3. compute the loss against `y_train`
4. zero gradients
5. call backward
6. call optimizer step

<details>
<summary>Hidden solution</summary>

```python
model_0.train()

y_pred = model_0(X_train)
loss = loss_fn(y_pred, y_train)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss)
```

</details>

---

## Exercise 8: Full Training Loop
Train the model for `100` epochs.  
Track:
- epoch number
- training loss
- test loss

For testing:
- switch to evaluation mode
- use `torch.inference_mode()`

<details>
<summary>Hidden solution</summary>

```python
torch.manual_seed(42)

model_0 = LinearRegressionModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

epochs = 100
epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    epoch_count.append(epoch)
    train_loss_values.append(loss.item())
    test_loss_values.append(test_loss.item())
```

</details>

---

## Exercise 9: Plot Loss Curves
Using the values collected in Exercise 8, plot:
- training loss over epochs
- test loss over epochs

<details>
<summary>Hidden solution</summary>

```python
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Curves")
plt.legend()
plt.show()
```

</details>

---

## Exercise 10: Compare Learned Parameters to True Parameters
After training, print:
- the learned weight and bias
- the original true weight and bias used to generate the data

<details>
<summary>Hidden solution</summary>

```python
print("Learned parameters:")
print(model_0.state_dict())

print("Original parameters:")
print(f"weight: {weight}, bias: {bias}")
```

</details>

---

## Exercise 11: Inference on Test Data
Set the trained model to evaluation mode and make predictions on `X_test` using `torch.inference_mode()`.

<details>
<summary>Hidden solution</summary>

```python
model_0.eval()

with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds)
```

</details>

---

## Exercise 12: Visualize Predictions
Modify your plotting function so it can also display model predictions on the test data.

<details>
<summary>Hidden solution</summary>

```python
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(train_data, train_labels, label="Train data")
    plt.scatter(test_data, test_labels, label="Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, label="Predictions")

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()

plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)
```

</details>

---

## Exercise 13: Save the Model
Save the trained model's `state_dict()` into a `models/` folder under the filename:

`01_pytorch_workflow_model_0.pth`

<details>
<summary>Hidden solution</summary>

```python
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(model_0.state_dict(), MODEL_SAVE_PATH)
print(MODEL_SAVE_PATH)
```

</details>

---

## Exercise 14: Load the Saved Model
Create a fresh model instance and load the saved `state_dict()` into it.

<details>
<summary>Hidden solution</summary>

```python
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))
```

</details>

---

## Exercise 15: Verify Loaded Model Predictions
Check whether the loaded model produces the same predictions as the original trained model on `X_test`.

<details>
<summary>Hidden solution</summary>

```python
loaded_model_0.eval()

with torch.inference_mode():
    loaded_preds = loaded_model_0(X_test)

print(torch.allclose(y_preds, loaded_preds))
```

</details>

---

## Exercise 16: Concept Check
Answer in words:

1. Why do we call `optimizer.zero_grad()`?
2. Why do we use `model.eval()` during testing?
3. Why is `torch.inference_mode()` helpful during inference?
4. Why do we usually save `state_dict()` instead of the full model object?

<details>
<summary>Hidden solution</summary>

```text
1. We call optimizer.zero_grad() so gradients from previous steps do not accumulate unintentionally.
2. model.eval() switches the model into evaluation behavior, which matters for layers like dropout and batch norm.
3. torch.inference_mode() disables gradient tracking, making inference faster and more memory efficient.
4. state_dict() is the standard portable way to save learned parameters without tightly coupling to the exact Python object.
```

</details>

---

## Exercise 17: Extension Challenge
Change the true underlying line to:

- `weight = -1.2`
- `bias = 0.9`

Regenerate the data and retrain the model.  
Does the model still learn the parameters well?

<details>
<summary>Hidden solution</summary>

```python
weight = -1.2
bias = 0.9

X = torch.arange(0, 1, 0.02)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

torch.manual_seed(42)
model_1 = LinearRegressionModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)

for epoch in range(100):
    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model_1.state_dict())
print(f"True weight: {weight}, True bias: {bias}")
```

</details>
