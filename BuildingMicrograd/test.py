
#from BuildingMicrograd.structures import Value, Neuron, Layer, NeuralNetwork
#from BuildingMicrograd.visualization_functions import *

from structures import Value
from visualization_functions import *

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b + c
d = e * 2.0
f = d.tanh()
f.backward()
print(f)

draw_dot(f)


