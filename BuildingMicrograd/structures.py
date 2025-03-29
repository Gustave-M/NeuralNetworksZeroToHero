import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Self

class Value:
    def __init__(self:Self, data, _children=(), _op='', label='')->None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self:Self)->None:
        return f"Value(data={self.data})"
    
    
    def __add__(self:Self, other)->Self:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = +1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self:Self, other)->Self:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self:Self, other)->Self:
        return self * other
    
    def __radd__(self:Self, other)->Self:
        return self + other
    
    def exp(self:Self)->Self:
        x = self.data
        out = Value(math.e**x, (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out.backward = _backward

        return out
    
    def __pow__(self:Self, other)->Self:
        assert isinstance(other, (int, float)), "Power must be int or float for the moment ..."
        out = Value(self.data**other, (self, ), f'** {other}')

        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self:Self, other)->Self:
        return self * other**-1
    
    def __neg__(self:Self)->Self:
        return self * -1

    def __sub__(self:Self, other:Self)->Self:
        return self + (-other)
    
    def tanh(self:Self)->Self:
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self:Self)->None:
        topo = []
        visited = set()
        def build_topo(v:Self):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

class Neuron:

    def __init__(self:Self, nin):
        # nin, number of inputs
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1))

    def __call__(self:Self, x):
        # w*x + b

        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self:Self):
        return self.w + [self.b]

class Layer:

    def __init__(self:Self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self:Self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self:Self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    
    def __init__(self:Self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self:Self, x):
        for layer in self.layers:
            x = layer(x) # Update au fil des couche les entrées (joli)
        return x
    
    def parameters(self:Self):
        return [p for layer in self.layers for p in layer.parameters()]


