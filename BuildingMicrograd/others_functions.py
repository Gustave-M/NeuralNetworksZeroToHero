import math
import numpy as np
from typing import Self

def loss(ys, ypred):
    return sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))


