
import numpy as np

N = np.array([[1, 2], [3, 4]])
N = N / N.sum(1 , keepdims=True)
print(N)
