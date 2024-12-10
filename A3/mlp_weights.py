# %% Use PyTorch for Backpropagation and add Biases and plot activations
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nn_tools import visualize_training

# TODO: Complete the modifications to this script according to the instructions in Q1.2c.
# - Keep versions of this file for reference as you work on changes.
# - Submit only one final version, the one you prefer for grading.
# - Please keep comments for deactivated code changes to document different versions.
# - Include plots from all versions you experimented with in your report.

# Initialize weights and inputs
W0 = torch.tensor([[0.2, 0.3], 
                   [0, -0.1]], requires_grad=True)
W1 = torch.tensor([[0.4, 0], 
                   [0, 0.6], 
                   [-0.2, 0.1]], requires_grad=True)
W2 = torch.tensor([0.3, 0.5, 0], requires_grad=True)

relu_type = F.sigmoid
x = torch.tensor([1.0, 1.0])
y_target = torch.tensor(1)
learning_rate = 0.1
epochs = 50

variables = [W0, W1, W2]

# Mask for learnable weights in W1
mask_W0 = torch.tensor([[1, 0], 
                        [0, 1]], dtype=torch.bool)
mask_W1 = torch.tensor([[0, 1], 
                        [1, 0], 
                        [1, 1]], dtype=torch.bool)
mask_W2 = torch.tensor([1, 0, 1], dtype=torch.bool)

# Initialize masked weights directly
use_zeros = True
if use_zeros:
    with torch.no_grad():
        W0[mask_W0] = torch.zeros(W0[mask_W0].size())
        W1[mask_W1] = torch.zeros(W1[mask_W1].size())
        W2[mask_W2] = torch.zeros(W2[mask_W2].size())
    print(f"W0 tunable elements: {W0[mask_W0.bool()]}")
    print(f"W1 tunable elements: {W1[mask_W1.bool()]}")
    print(f"W2 tunable elements: {W2[mask_W2.bool()]}")

# Define a forward pass function
def forward_pass(x, W0, W1, W2, b0=1, b1=1, b2=1, relu=F.relu):
    z1 = torch.matmul(W0, x) + (b0 if b0 is not None else 0)
    h1 = relu(z1)
    z2 = torch.matmul(W1, h1) + (b1 if b1 is not None else 0)
    h2 = relu(z2)
    y_pred = torch.matmul(W2, h2) + (b2 if b2 is not None else 0)
    return z1, h1, z2, h2, y_pred

# Initialize optimizer with weights and biases
optimizer = torch.optim.SGD(variables, lr=learning_rate)

losses = []
parameter_evolution = {"W0": [], "W1": [], "W2": []}

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    z1, h1, z2, h2, y_pred = forward_pass(x, W0, W1, W2, relu=relu_type)
    loss = (y_pred - y_target) ** 2
    loss.backward()
    
    # Apply mask to freeze non-learnable weights in W1
    with torch.no_grad():
        W0.grad *= mask_W0
        W1.grad *= mask_W1
        W2.grad *= mask_W2

    optimizer.step()

    # Log loss and parameter values
    losses.append(loss.item())
    parameter_evolution["W0"].append(W0.detach().clone())
    parameter_evolution["W1"].append(W1.detach().clone())
    parameter_evolution["W2"].append(W2.detach().clone())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}, y_pred = {y_pred.item()}")

# Visualize results, you can also save these plots to disk
epochs_range = range(epochs)
visualize_training(epochs_range, losses, parameter_evolution,
                   save_filename="test", save_format='png'
                   )

#%% Generate LaTeX code for the final weights
## Auto-generating tex from python
#from nn_tools import generate_tex_code
#print(generate_tex_code("changed_something"))

# %%

print(f"W0 elements tuned: {W0[mask_W0.bool()]}")
print(f"W1 elements tuned: {W1[mask_W1.bool()]}")
print(f"W2 elements tuned: {W2[mask_W2.bool()]}")
