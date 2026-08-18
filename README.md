# README — Manual Neural Network from Scratch Using PyTorch

````markdown
# Simple Neural Network from Scratch Using PyTorch

A simple binary classification neural network implemented **from scratch using PyTorch**, without using PyTorch's built-in `nn.Module`, optimizer, or loss functions.

This project focuses on understanding the fundamental concepts behind neural network training, including **forward propagation, Binary Cross-Entropy loss, backpropagation, gradient descent, and parameter updates**.

## 📌 Project Overview

The neural network consists of:

- Input features
- Trainable weights
- Trainable bias
- Sigmoid activation function
- Binary Cross-Entropy (BCE) loss
- Backpropagation using PyTorch Autograd
- Manual gradient descent parameter updates

The main purpose of this project is to understand **what happens internally during neural network training** rather than relying on high-level PyTorch APIs.

---

## 🧠 Model Architecture

The implemented model follows this structure:

```text
Input Features (X)
        │
        ▼
   Linear Layer
   z = XW + b
        │
        ▼
 Sigmoid Activation
   ŷ = σ(z)
        │
        ▼
   Prediction (ŷ)
        │
        ▼
 Binary Cross-Entropy
        │
        ▼
      Loss
````

Since the model contains only a single linear layer followed by a sigmoid activation, it is mathematically equivalent to **Logistic Regression**.

---

## 🔢 Mathematical Formulation

### 1. Linear Transformation

The model first calculates:

$$
z = XW + b
$$

where:

* $X$ = input features
* $W$ = trainable weights
* $b$ = bias
* $z$ = linear output

### 2. Sigmoid Activation

The prediction is calculated using:

$$
\hat{y} = \sigma(z)
$$

where:

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

The sigmoid function converts the output into a probability between 0 and 1.

### 3. Binary Cross-Entropy Loss

The loss function used is:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
$$

A small epsilon value is used to prevent numerical problems caused by $\log(0)$.

---

## 🔄 Training Process

The model is trained using the following sequence:

```text
1. Initialize weights and bias
          ↓
2. Forward Pass
          ↓
3. Calculate Prediction
          ↓
4. Calculate BCE Loss
          ↓
5. Backpropagation
          ↓
6. Calculate Gradients
          ↓
7. Update Weights and Bias
          ↓
8. Reset Gradients
          ↓
9. Repeat for each epoch
```

---

## 🔙 Backpropagation

Instead of manually deriving every gradient, PyTorch's **Autograd** system is used to calculate the gradients.

The parameters are initialized with:

```python
requires_grad=True
```

After calculating the loss:

```python
loss.backward()
```

PyTorch automatically calculates:

```text
∂Loss/∂Weights
∂Loss/∂Bias
```

These gradients are stored in:

```python
model.weights.grad
model.bias.grad
```

---

## 📉 Gradient Descent

The parameters are updated manually using:

$$
W_{new} = W_{old} - \alpha\frac{\partial L}{\partial W}
$$

$$
b_{new} = b_{old} - \alpha\frac{\partial L}{\partial b}
$$

where $\alpha$ is the learning rate.

In the implementation:

```python
with torch.no_grad():
    model.weights -= learning_rate * model.weights.grad
    model.bias -= learning_rate * model.bias.grad
```

Gradients are reset after every update:

```python
model.weights.grad.zero_()
model.bias.grad.zero_()
```

This is necessary because PyTorch accumulates gradients by default.

---

## 🏗️ Implementation

### Model Class

```python
class Simple_NN():

    def __init__(self, x):

        self.weights = torch.rand(
            x.shape[1],
            1,
            dtype=torch.float64,
            requires_grad=True
        )

        self.bias = torch.zeros(
            1,
            dtype=torch.float64,
            requires_grad=True
        )

    def forward_pass(self, X):

        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)

        return y_pred

    def loss_function(self, y_pred, y):

        epsilon = 1e-7

        y_pred = torch.clamp(
            y_pred,
            epsilon,
            1 - epsilon
        )

        loss = -(y * torch.log(y_pred) +
                 (1 - y) * torch.log(1 - y_pred)).mean()

        return loss
```

---

## 🚀 Training Loop

The model is trained manually using the following loop:

```python
model = Simple_NN(X_train_tensor)

epoch = 25
learning_rate = 0.1

for i in range(epoch):

    # Forward Pass
    y_pred = model.forward_pass(X_train_tensor)

    # Loss Calculation
    loss = model.loss_function(
        y_pred,
        y_train_tensor
    )

    # Backward Pass
    loss.backward()

    # Parameter Update
    with torch.no_grad():

        model.weights -= (
            learning_rate * model.weights.grad
        )

        model.bias -= (
            learning_rate * model.bias.grad
        )

    # Reset Gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Jupyter Notebook / Google Colab

---

## 📂 Project Structure

```text
simple-neural-network-from-scratch/
│
├── README.md
├── neural_network_from_scratch.ipynb
│
└── dataset/
    └── dataset.csv
```

> Dataset files can be included depending on the dataset used in the notebook.

---

## 🎯 Learning Objectives

Through this project, I explored:

* How a neural network performs a forward pass
* How weights and bias are initialized
* How the sigmoid activation function works
* How Binary Cross-Entropy is calculated
* Why numerical stability is important
* How `requires_grad=True` works
* How PyTorch Autograd builds a computation graph
* How `loss.backward()` calculates gradients
* How gradients are stored in `.grad`
* How gradient descent updates model parameters
* Why gradients need to be reset after each iteration
* The relationship between a single-layer neural network and logistic regression

---

## 🔍 Key Takeaway

This project was implemented without using:

```python
torch.nn.Module
torch.optim
torch.nn.BCELoss
```

Instead, the core training process was implemented manually to understand the mechanics behind neural network learning.

The only major automatic component used is **PyTorch Autograd for gradient calculation**.

---

## 📌 Future Improvements

Possible extensions of this project include:

* Adding multiple hidden layers
* Implementing ReLU and other activation functions
* Implementing different loss functions
* Writing manual backpropagation without Autograd
* Implementing a custom optimizer
* Adding mini-batch gradient descent
* Adding accuracy, precision, recall, and F1-score
* Visualizing the training loss
* Comparing the implementation with PyTorch's `nn.Module`
* Extending the model to multiclass classification

---

## 👨‍💻 Author

**S. M. Anik Hasan**

Electrical & Electronic Engineering Student
Interested in Power Electronics, Machine Learning, and Artificial Intelligence.

---

## ⭐ Acknowledgment

This project was developed as a learning exercise to understand the fundamental mathematics and implementation details behind neural network training using PyTorch.

```
```
