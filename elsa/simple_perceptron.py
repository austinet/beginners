# Simple Perceptron #
# Available on Numpy #
# Austin Hyeon, 2019. "Sociology meets Computer Science" #

import numpy as np


def LOGIC_GATE(list, weight, bias):
    '''
    Parameters
    ---
        array: One dimensional array [n].
        W: Weight
        b: bias

    Reference for binary gates
    ---
        AND Gate: weight=0.5, bias=-0.7
        NAND Gate: weight=-0.5, bias=0.7
        OR Gate: weight=0.5, bias=-0.2
        NOR Gate: weight=-0.5, bias=0.2
    '''

    x = np.array(list)
    W = np.array([weight, weight])
    b = bias

    input_tensor = np.sum(x * W) + b

    # An activation function for the simple perceptron
    # The name is "Step Function"
    if input_tensor <= 0: return 0
    else: return 1


def AND(list):
    '''
    Only available on the binary system
    '''
    output_tensor = LOGIC_GATE(list, 0.5, -0.7)
    return output_tensor


def OR(list):
    '''
    Only available on the binary system
    '''
    output_tensor = LOGIC_GATE(list, 0.5, -0.2)
    return output_tensor


def NAND(list):
    '''
    Only available on the binary system
    '''
    output_tensor = LOGIC_GATE(list, -0.5, 0.7)
    return output_tensor


def NOR(list):
    '''
    Only available on the binary system
    '''
    output_tensor = LOGIC_GATE(list, -0.5, 0.2)
    return output_tensor
    

def XOR(list):
    '''
    Only available on the binary system
    '''
    s1 = NAND(list)
    s2 = OR(list)
    output_tensor = AND([s1, s2])
    return output_tensor


def XNOR(list):
    '''
    Only available on the binary system
    '''
    s1 = NAND(list)
    s2 = OR(list)
    output_tensor = XOR([s1, s2])
    return output_tensor


binary = [[0, 0], [0, 1], [1, 0], [1, 1]]

print('AND')
for i in range(len(binary)):
    value = AND(binary[i])
    print(value)

print('OR')
for i in range(len(binary)):
    value = OR(binary[i])
    print(value)

print('NAND')
for i in range(len(binary)):
    value = NAND(binary[i])
    print(value)

print('NOR')
for i in range(len(binary)):
    value = NOR(binary[i])
    print(value)

print('XOR')
for i in range(len(binary)):
    value = XOR(binary[i])
    print(value)

print('XNOR')
for i in range(len(binary)):
    value = XNOR(binary[i])
    print(value)