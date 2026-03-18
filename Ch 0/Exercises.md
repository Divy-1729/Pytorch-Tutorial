# PyTorch Fundamentals Exercises

These exercises are based on the concepts covered in `C0.ipynb`.  
Each solution is hidden inside a collapsible section so you can attempt first. I highly recommend attempting and not looking at the solution directly.

---

## Exercise 1: Tensor Dimensions
Create:
- a scalar tensor  
- a vector tensor  
- a matrix tensor  
- a 3D tensor  

Then print `.ndim` for each.

<details>
<summary>Solution</summary>

```python
import torch

scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor3d = torch.tensor([[[1, 2], [3, 4]]])

print(scalar.ndim)
print(vector.ndim)
print(matrix.ndim)
print(tensor3d.ndim)
```

</details>

---

## Exercise 2: Shape vs Number of Dimensions
Construct a tensor of shape `(2, 3, 4)` and verify:
- its shape  
- its number of dimensions  

<details>
<summary>Solution</summary>

```python
x = torch.rand(2, 3, 4)
print(x.shape)
print(x.ndim)
```

</details>

---

## Exercise 3: Random Tensor Initialization
Generate:
- a random tensor of shape `(3, 4)`
- a tensor of zeros with shape `(2, 5)`
- a tensor of ones with shape `(4,)`

<details>
<summary>Hidden solution</summary>

```python
rand_tensor = torch.rand(3, 4)
zeros = torch.zeros(2, 5)
ones = torch.ones(4)

print(rand_tensor)
print(zeros)
print(ones)
```

</details>

---

## Exercise 4: Tensor Datatypes
Create one tensor as `float32` and another as `int64`.  
Check their `.dtype`.

<details>
<summary>Solution</summary>

```python
a = torch.tensor([1.0, 2.0], dtype=torch.float32)
b = torch.tensor([1, 2], dtype=torch.int64)

print(a.dtype)
print(b.dtype)
```

</details>

---

## Exercise 5: Basic Tensor Arithmetic
Let:

```python
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
```

Compute:
- addition  
- subtraction  
- elementwise multiplication  

<details>
<summary>Solution</summary>

```python
print(a + b)
print(a - b)
print(a * b)
```

</details>

---

## Exercise 6: Matrix Multiplication
Construct two tensors with compatible dimensions and compute matrix multiplication using:
- `torch.matmul`
- `@`

<details>
<summary>Solution</summary>

```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

print(torch.matmul(A, B))
print(A @ B)
```

</details>

---

## Exercise 7: Reshaping
Start with:

```python
x = torch.arange(12)
```

Convert it into:
- shape `(3,4)`
- shape `(2,6)`

<details>
<summary>Solution</summary>

```python
x = torch.arange(12)

print(x.reshape(3,4))
print(x.reshape(2,6))
```

</details>

---

## Exercise 8: Stack vs Squeeze
Take:

```python
x = torch.tensor([[1,2,3]])
```

- stack two copies along dimension 0  
- squeeze the original tensor  

<details>
<summary>Solution</summary>

```python
stacked = torch.stack([x, x], dim=0)
squeezed = x.squeeze()

print(stacked.shape)
print(squeezed.shape)
```

</details>

---

## Exercise 9: Indexing Practice
Given:

```python
x = torch.arange(24).reshape(2,3,4)
```

Extract:
- the first matrix  
- the second row of the first matrix  
- the final element  

<details>
<summary>Solution</summary>

```python
print(x[0])
print(x[0,1])
print(x[-1,-1,-1])
```

</details>

---

## Exercise 10: Reproducibility
Set a manual seed and generate two identical random tensors.

<details>
<summary>Solution</summary>

```python
torch.manual_seed(42)
a = torch.rand(3)

torch.manual_seed(42)
b = torch.rand(3)

print(a)
print(b)
```

</details>

---

## Exercise 11: Device Awareness
Write code that:
- checks whether CUDA is available  
- places a tensor on GPU if available  

<details>
<summary>Solution</summary>

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([1,2,3]).to(device)

print(device)
print(x)
```

</details>

---

## Exercise 12: Challenge Exercise
Without running code first, predict the output shape:

```python
x = torch.rand(2,3,4)
y = x.permute(2,0,1)
```

Then verify.

<details>
<summary>Hidden solution</summary>

```python
x = torch.rand(2,3,4)
y = x.permute(2,0,1)

print(y.shape)   # (4,2,3)
```

</details>
