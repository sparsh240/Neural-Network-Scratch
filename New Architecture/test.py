from functional_3_complete import *

import numpy as np
import matplotlib.pyplot as plt

# Generate a Spiral Dataset 
N = 400 # Points per class
X = np.zeros((N*2, 2))
y = np.zeros(N*2)

for j in range(2):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

plt.figure(figsize=(6, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', s=20, label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', s=20, label='Class 1')
plt.title("Complex Non-Linear Data (Spirals)")
plt.legend()
plt.show()

# Network
nn = Network(loss_fn='bceloss')
nn.Layer(2, 32, 'relu')
nn.Layer(32, 32, 'relu')
nn.Layer(32, 1, 'sigmoid')

epochs = 100
lr = 0.03

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    
    # Shuffle data for better SGD convergence
    indices = np.random.permutation(len(X))
    
    for i in indices:
        x_i = X[i]
        y_i = np.array([y[i]])
        
        pred = nn.forward(x_i)
        loss = nn.backward(y_i, pred, lr)
        
        total_loss += loss
        if (pred[0] >= 0.5) == y_i[0]:
            correct += 1
            
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch: {epoch+1:03d}/{epochs} | Loss: {total_loss/len(X):.4f} | Accuracy: {correct/len(X):.4f}")

nn.mode = 'test'

h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid = np.c_[xx.ravel(), yy.ravel()]
preds = np.array([nn.forward(g)[0] for g in grid]).reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], colors=['#ff9999', '#9999ff'], alpha=0.5)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', edgecolors='k', s=20)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', edgecolors='k', s=20)
plt.title("Model Decision Boundary on Spirals")
plt.show()