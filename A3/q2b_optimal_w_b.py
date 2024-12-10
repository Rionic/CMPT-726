from sklearn.svm import SVC
import numpy as np

X = np.array([[2, 6], [6, 2], [4, 6], [5, 3]])
y = np.array([1, -1, 1, -1])

model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

w_star = model.coef_[0]
b_star = model.intercept_[0]
print('w*:', w_star)
print('b*:', b_star)