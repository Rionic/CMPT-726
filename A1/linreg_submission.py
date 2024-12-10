# %% [markdown]
# 

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error
# Load Boston Housing Data from GitHub
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
# Features: ’rm’ (number of rooms) and ’lstat’ (lower status population, percentage)
X = df[['rm', 'lstat']]
y = df['medv'] # Target: ’medv’ (median house value in $1,000s)



# %%
best_degree_linear = None
best_degree_ridge = None
best_alpha_ridge = None
min_mse_linear = float('inf')
min_mse_ridge = float('inf')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_model = LinearRegression()
ridge_model = Ridge()
for i in range(1, 6):
    poly = PolynomialFeatures(i)
    X_train_poly = poly.fit_transform(X_train)
    ridge_grid = { 'alpha': [0.1, 1, 10, 100] }

    mse_score_linear = cross_val_score(linear_model, X_train_poly, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_score_linear = -mse_score_linear
    mse_score_linear_avg = mse_score_linear.mean()

    grid_search = GridSearchCV(estimator=ridge_model, param_grid=ridge_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train_poly, y_train)
    
    ridge_mse = -grid_search.best_score_
    ridge_alpha = grid_search.best_params_['alpha']
    
    if mse_score_linear_avg < min_mse_linear:
        min_mse_linear = mse_score_linear_avg
        best_degree_linear = i

    if ridge_mse < min_mse_ridge:
        min_mse_ridge = ridge_mse
        best_degree_ridge = i
        best_alpha_ridge = ridge_alpha

    print(f'Degree {i}')
    print('Linear Regression average MSE:', mse_score_linear_avg)
    print('Ridge Regression average MSE:', ridge_mse)
    print('Ridge Regression best alpha:', ridge_alpha, '\n')
    
print('Overall best parameters')
print('Linear Regression best degree:', best_degree_linear)
print('Ridge Regression best degree:', best_degree_ridge)
print('Ridge Regression best alpha:', best_alpha_ridge)


# %%
# Now we will train and test the models
poly_best = PolynomialFeatures(best_degree_ridge)

X_train_poly_best = poly_best.fit_transform(X_train)

linear_model_best = LinearRegression().fit(X_train_poly_best, y_train)
ridge_model_best = Ridge(alpha=best_alpha_ridge).fit(X_train_poly_best, y_train)

X_test_poly = poly_best.transform(X_test)

y_pred_linear = linear_model_best.predict(X_test_poly)
y_pred_ridge = ridge_model_best.predict(X_test_poly)

# MSE function is deprecated, so we use RMSE and square the score at the end
mse_linear = root_mean_squared_error(y_test, y_pred_linear)
mse_ridge = root_mean_squared_error(y_test, y_pred_ridge)
print('Linear Regression error:', mse_linear**2)
print('Ridge Regression error:', mse_ridge**2)

# %%
"""
Questions

Part C
Ridge regression has a better MSE score, meaning in this case, it generalizes to new data better. In general, Ridge
regression is less prone to overfitting due to penalization of high degrees, this means that for general datasets, 
ridge regression will usually generalize to new data better.

Part D
High degree polynomials can overfit due to their flexibilty. Ridge penalizes this by adding a regularization term
(alpha) which is proportional to the sum of the squared coefficients. The larger the alpha & degree, the stronger
the penalization. The choice of alpha can change how closely the model fits the data. A large degree can lead to
overfitting, and a large alpha can lead to underfitting. The overfitting can be countered by a large enough alpha.
"""


