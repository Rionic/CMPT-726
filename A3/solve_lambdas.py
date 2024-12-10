#%%
import numpy as np
from scipy.optimize import minimize

#%% Step 1: Define the data points and labels
X = np.array([[2, 6], [6, 2], [4, 6], [5, 3]])
y = np.array([1, -1, 1, -1])

# Number of data points
N = len(y)

#%% Step 2: Compute the Gram matrix K
# TODO: compute K
K = np.zeros((N, N))

K = np.dot(X, X.T)

#%% Step 3: Set up and solve the quadratic programming problem for lambda using scipy.optimize
# Define the dual objective function for minimization (note we minimize the negative for maximization)
def dual_objective(lambda_):
    first_term = -np.sum(lambda_)
    second_term = 0.5 * np.sum([lambda_[i] * lambda_[j] * y[i] * y[j] * K[i, j] for i in range(N) for j in range(N)])
    return first_term + second_term

# Set up the constraints and bounds
constraints = {'type': 'eq', 'fun': lambda lambda_: np.dot(lambda_, y)}
C = 1
bounds = [(0, C) for _ in range(N)]

# Solve for lambda using scipy's minimize with SLSQP
initial_lambda = np.zeros(N)
solution = minimize(dual_objective, initial_lambda, bounds=bounds, constraints=constraints, method='SLSQP')

# Extract lambda values from the solution
lambda_ = solution.x

# Display the resulting lambda values
print("Optimal lambda values:", lambda_)

#%% Step 4: Identify the support vectors (lambda_i > 0)
support_vectors = [i for i in range(N) if lambda_[i] > 1e-5]
print("Support vector indices:", support_vectors)

#%% Step 5: Verification (Calculate w* and decision boundary)
w_star = sum(lambda_[i] * y[i] * X[i] for i in range(N))
print("Optimal weight vector w*:", w_star)

b_star_candidates = [y[i] - np.dot(w_star, X[i]) for i in support_vectors]
b_star = np.mean(b_star_candidates)
print("Optimal bias b*:", b_star)
