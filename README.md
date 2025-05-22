<H3>Name: Vishal S</H3>
<H3>Register No. 212223110063 </H3>
<H3>EX. NO.5</H3>
<H3>DATE:</H3>

<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>

<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

# Program:
## 1.Importing packages:
```
import numpy as np
import matplotlib.pyplot as plt
```
## 2.Guassian RBF function:
```
def gaussian_rbf(x, center, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - center)**2)
```
## 3.RBF feature declaration:
```
def build_rbf_features(X, centers, gamma=1):
    features = []
    for x in X:
        rbf_vals = [gaussian_rbf(x, c, gamma) for c in centers]
        rbf_vals.append(1)
        features.append(rbf_vals)
    return np.array(features)
```
## 4.RBF training:
```
def train_rbf(X, y, centers, gamma=1):
    Phi = build_rbf_features(X, centers, gamma)
    weights = np.linalg.pinv(Phi).dot(y)
    return weights
```
## 5.Prediction:
```
def predict_rbf(X, centers, weights, gamma=1):
    Phi = build_rbf_features(X, centers, gamma)
    return np.round(Phi.dot(weights))
```
## 6.Input dataset:
```
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

centers = np.array([[0,1], [1,0]])

weights = train_rbf(X, y, centers, gamma=1)

predictions = predict_rbf(X, centers, weights, gamma=1)
```
## 7.Plotting of Input space and Hidden space of RBF:
```
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o', s=100)
plt.title("Input Space (XOR)", fontsize=14)
plt.xlabel("$X1$", fontsize=12)
plt.ylabel("$X2$", fontsize=12)
plt.grid(True)

phi_X = build_rbf_features(X, centers, gamma=1)[:, :-1]
plt.subplot(1, 2, 2)
plt.scatter(phi_X[:, 0], phi_X[:, 1], c=y, cmap='coolwarm', marker='o', s=100)
plt.title("Hidden Space (RBF-transformed)", fontsize=14)
plt.xlabel("$\phi_1$", fontsize=12)
plt.ylabel("$\phi_2$", fontsize=12)
plt.grid(True)

plt.show()
```
# Output:
![image](https://github.com/user-attachments/assets/4a2c022a-d955-4127-9a4b-2c0e1be08428)
![image](https://github.com/user-attachments/assets/e5075f79-1d27-4acc-84ee-496d5db137be)
![image](https://github.com/user-attachments/assets/2f9e40be-3f91-4676-9b15-6673338f3c6d)
![image](https://github.com/user-attachments/assets/8d26e052-ef1c-4b23-a294-94a162cf0084)

<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.








