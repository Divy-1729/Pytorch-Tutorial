# Chapter 2 exercises.

These exercises are based on the concepts covered in  `C3.ipynb`.  
Each solution is hidden inside a collapsible section so you can try the exercise first.

---

## Exercise 1: Generate Circle Data
Use `make_circles` from `sklearn.datasets` to generate a binary classification dataset with:
- `n_samples = 1000`
- `noise = 0.03`
- `random_state = 42`

Then print the first 5 feature rows and first 5 labels.

<details>
<summary>Hidden solution</summary>

```python
from sklearn.datasets import make_circles

n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

print("X data:", X[:5])
print("y data:", y[:5])
```

</details>

---

## Exercise 2: Put the Data in a DataFrame
Create a pandas DataFrame with columns:
- `X1`
- `X2`
- `label`

Then display the first 10 rows.

<details>
<summary>Hidden solution</summary>

```python
import pandas as pd

circles = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})

circles.head(10)
```

</details>

---

## Exercise 3: Check Class Balance
Count how many examples belong to each class.

<details>
<summary>Hidden solution</summary>

```python
circles.label.value_counts()
```

</details>

---

## Exercise 4: Visualize the Dataset
Create a scatter plot of the circle data where:
- `X[:, 0]` is on the x-axis
- `X[:, 1]` is on the y-axis
- the color corresponds to the class label

<details>
<summary>Hidden solution</summary>

```python
import matplotlib.pyplot as plt

plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()
```

</details>

---

## Exercise 5: Convert to Tensors
Convert `X` and `y` from NumPy arrays to PyTorch tensors of type `float32`.

<details>
<summary>Hidden solution</summary>

```python
import torch

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5])
print(y[:5])
```

</details>

---

## Exercise 6: Create a Train/Test Split
Use `train_test_split` so that:
- 80% of the data is used for training
- 20% of the data is used for testing
- `random_state = 42`

Then print the lengths of each split.

<details>
<summary>Hidden solution</summary>

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(len(X_train), len(X_test), len(y_train), len(y_test))
```

</details>

---

## Exercise 7: Choose a Device
Write device-agnostic PyTorch code that uses CUDA if available, otherwise CPU.

<details>
<summary>Hidden solution</summary>

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```

</details>

---

## Exercise 8: Write ReLU and Sigmoid by Hand
Implement:
- `relu(x)`
- `sigmoid(x)`

using PyTorch operations only.

<details>
<summary>Hidden solution</summary>

```python
def relu(x: torch.Tensor):
    return torch.maximum(torch.zeros_like(x), x)

def sigmoid(x: torch.Tensor):
    return 1 / (1 + torch.exp(-x))
```

</details>

---

## Exercise 9: Build the Neural Network
Create a model class called `Hyperplanefinder` that:
- subclasses `nn.Module`
- takes in 2 input features
- has two hidden layers of size 10
- outputs 1 value
- uses ReLU activations between layers

<details>
<summary>Hidden solution</summary>

```python
from torch import nn

class Hyperplanefinder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(relu(self.layer_2(relu(self.layer_1(x)))))
```

</details>

---

## Exercise 10: Create the Model
Instantiate your model and move it to the target device.

<details>
<summary>Hidden solution</summary>

```python
model_0 = Hyperplanefinder().to(device)
model_0
```

</details>

---

## Exercise 11: Make Untrained Predictions
Pass `X_test` through the untrained model and inspect:
- the length of predictions
- prediction shape
- the first 10 predictions

<details>
<summary>Hidden solution</summary>

```python
untrained_preds = model_0(X_test.to(device))

print(f"Length of predictions: {len(untrained_preds)}")
print(f"Shape of predictions: {untrained_preds.shape}")
print(untrained_preds[:10])
```

</details>

---

## Exercise 12: Define Loss and Optimizer
For binary classification with logits:
- use `nn.BCEWithLogitsLoss()`
- use Adam with learning rate `0.001`

<details>
<summary>Hidden solution</summary>

```python
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)
```

</details>

---

## Exercise 13: Write an Accuracy Function
Write a function `accuracy_fn(y_true, y_pred)` that returns classification accuracy as a percentage.

<details>
<summary>Hidden solution</summary>

```python
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
```

</details>

---

## Exercise 14: Write Precision and Recall Functions
Implement:
- `precision_fn(y_true, y_pred)`
- `recall_fn(y_true, y_pred)`

for binary classification.

<details>
<summary>Hidden solution</summary>

```python
def recall_fn(y_true, y_pred):
    tp, fn = 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def precision_fn(y_true, y_pred):
    tp, fp = 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)
```

</details>

---

## Exercise 15: One Training Step
Write code for a single training step:
1. put the model in train mode
2. move training data to the device
3. compute logits
4. compute BCEWithLogits loss
5. zero gradients
6. backpropagate
7. update parameters

Remember to match shapes properly.

<details>
<summary>Hidden solution</summary>

```python
model_0.train()

X_train_on_device = X_train.to(device)
y_train_on_device = y_train.to(device)

y_pred_logits = model_0(X_train_on_device)
loss = loss_fn(y_pred_logits, y_train_on_device.unsqueeze(1))

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss.item())
```

</details>

---

## Exercise 16: Why `unsqueeze(1)`?
Suppose `y_pred_logits.shape == [800, 1]` but `y_train.shape == [800]`.

Explain why `y_train.unsqueeze(1)` is needed before computing `BCEWithLogitsLoss`.

<details>
<summary>Hidden solution</summary>

```text
BCEWithLogitsLoss expects the predictions and targets to have the same shape. 
The model outputs a column vector of shape [batch_size, 1], while y_train is a 1D tensor of shape [batch_size]. 
Using unsqueeze(1) changes the target shape to [batch_size, 1], so the loss can be computed correctly.
```

</details>

---

## Exercise 17: One Evaluation Step
Write code to:
- switch the model to eval mode
- use `torch.inference_mode()`
- compute test logits
- compute test loss

<details>
<summary>Hidden solution</summary>

```python
model_0.eval()

with torch.inference_mode():
    X_test_on_device = X_test.to(device)
    y_test_on_device = y_test.to(device)

    test_pred_logits = model_0(X_test_on_device)
    test_loss = loss_fn(test_pred_logits, y_test_on_device.unsqueeze(1))

print(test_loss.item())
```

</details>

---

## Exercise 18: Convert Logits to Class Predictions
After getting logits from the model:
1. apply sigmoid to get probabilities
2. round the probabilities to get predicted labels

<details>
<summary>Hidden solution</summary>

```python
with torch.inference_mode():
    logits = model_0(X_test.to(device))

probs = torch.sigmoid(logits)
preds = torch.round(probs)

print(probs[:10])
print(preds[:10])
```

</details>

---

## Exercise 19: Full Training Loop
Train the model for `1000` epochs and store:
- training loss
- test loss
- training accuracy
- test accuracy
- epoch count

Only record and print metrics every 10 epochs.

<details>
<summary>Hidden solution</summary>

```python
n_epochs = 1000

train_loss_values = []
test_loss_values = []
train_acc_values = []
test_acc_values = []
epoch_count = []

for epoch in range(n_epochs):
    model_0.train()

    X_train_on_device = X_train.to(device)
    y_train_on_device = y_train.to(device)

    y_pred_logits = model_0(X_train_on_device)
    loss = loss_fn(y_pred_logits, y_train_on_device.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        X_test_on_device = X_test.to(device)
        y_test_on_device = y_test.to(device)

        test_logits = model_0(X_test_on_device)
        test_loss = loss_fn(test_logits, y_test_on_device.unsqueeze(1))

    if epoch % 10 == 0:
        train_preds = torch.round(torch.sigmoid(y_pred_logits)).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits)).squeeze()

        train_acc = accuracy_fn(y_train_on_device, train_preds)
        test_acc = accuracy_fn(y_test_on_device, test_preds)

        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        train_acc_values.append(train_acc)
        test_acc_values.append(test_acc)

        print(
            f"Epoch: {epoch} | "
            f"Train loss: {loss:.5f} | Test loss: {test_loss:.5f} | "
            f"Train acc: {train_acc:.2f}% | Test acc: {test_acc:.2f}%"
        )
```

</details>

---

## Exercise 20: Plot Loss Curves
Using the lists collected during training, plot:
- train loss vs epoch
- test loss vs epoch

<details>
<summary>Hidden solution</summary>

```python
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

</details>

---

## Exercise 21: Plot Accuracy Curves
Using the stored values, plot:
- train accuracy vs epoch
- test accuracy vs epoch

<details>
<summary>Hidden solution</summary>

```python
plt.plot(epoch_count, train_acc_values, label="Train accuracy")
plt.plot(epoch_count, test_acc_values, label="Test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()
```

</details>

---

## Exercise 22: Build the Same Idea with `nn.Sequential`
Rebuild a smaller version of the classifier using `nn.Sequential`.

<details>
<summary>Hidden solution</summary>

```python
model_seq = nn.Sequential(
    nn.Linear(in_features=2, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=1)
).to(device)

model_seq
```

</details>

---

## Exercise 23: Concept Check
Answer the following in words:

1. Why is this a binary classification task?
2. Why do we use `BCEWithLogitsLoss` instead of `BCELoss` here?
3. Why do we apply `sigmoid` only when converting logits into probabilities?
4. Why can a simple linear boundary struggle on circle data?

<details>
<summary>Hidden solution</summary>

```text
1. It is binary classification because each point belongs to one of two classes: 0 or 1.
2. BCEWithLogitsLoss is numerically more stable because it combines the sigmoid step and BCE loss into one function.
3. The model outputs raw logits during training, and sigmoid converts those logits into probabilities between 0 and 1 when needed for interpretation.
4. Circle data is not linearly separable in the original 2D space, so the model needs nonlinear transformations to learn a curved decision boundary.
```

</details>

---

## Exercise 24: Extension Challenge
Try changing one of the following and retraining:
- hidden layer width from 10 to 32
- learning rate from `0.001` to `0.01`
- optimizer from Adam to SGD

Write 2–3 sentences describing how the learning behavior changes.

<details>
<summary>Hidden solution</summary>

```text
Possible observation:
Increasing hidden width can help the model represent a more flexible nonlinear boundary.
A larger learning rate may train faster at first but can also become unstable.
Switching from Adam to SGD may slow training unless the learning rate is tuned carefully.
```

</details>
