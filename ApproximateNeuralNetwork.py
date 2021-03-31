import numpy as np
import math

import tensorflow as tf
from tensorflow import keras

from keras import Model
from keras.layers import Dense, Dropout, Input

class Error(Exception):
    pass

class LayerNotSupported(Error):
    pass

class ApproximateNeuralNetwork:
    def __init__(self, model, quantization_precision = 100, mult_wd = 3, nofbits = 8):
        self.quantization_precision = quantization_precision
        self.mult_wd = mult_wd
        self.nofbits = nofbits

        self.model = model
        self.barebones_model = getBareBonesModel(model, quantization_precision)

    def predict(self, inp):
        curr = inp
        for (layerName, weights, bias) in self.barebones_model:
            if layerName=="dense":
                curr = dense(curr, weights, bias, self.nofbits, self.mult_wd)
            elif layerName=="conv2d":
                curr = conv2d(curr, weights, bias, self.nofbits, self.mult_wd)
            elif layerName=="flatten":
                curr = curr.flatten()
        return curr

    def setMultWD(self, value):
        self.mult_wd = value
    
    def setNumBits(self, value):
        self.nofbits = value

    def setQuantizationPrecision(self, value):
        self.quantization_precision = value
        self.barebones_model = getBareBonesModel(self.model, value)

    def summary(self):
        print("topology")
        for (name, _, _) in self.barebones_model:
            print(name)
        print()
        print("Quantization precision:", self.quantization_precision)
        print("Number of bits:", self.nofbits)
        print("Multiply width:", self.mult_wd)



def to_nofbits(int_num, nofbits):
    maximum = pow(2,nofbits-1)
    if(int_num<0):
        return min(abs(int_num), maximum)*-1
    return min(int_num, maximum)

def twos_complement(val,nofbits):
    if val < 0:
        y = abs(val)
        negcheck = True         
    else: 
        y = val                 
        negcheck = False        
    return y, negcheck

#Dynamic sliding multiplexor
def multiplixing(var,log_val,mult_wd):
    masking_var = 0
    if (log_val > (mult_wd -1)):   
        for n in range (log_val, log_val - mult_wd,-1):       
            masking_var = pow(2,n) + masking_var
        var = var & masking_var
        return var
    else:
        return var



def shamt(log_value, mult_wd):
    if log_value > (mult_wd -1):
        shamt = log_value - (mult_wd -1) 
    else: 
        shamt = 0
    return shamt

def multiplyNumbers(a,b,nofbits,mult_wd):
    a = int(a)
    b = int(b)

    a = to_nofbits(a, nofbits)
    b = to_nofbits(b, nofbits)

    a_y, sign_a = twos_complement(a,nofbits)
    b_y, sign_b = twos_complement(b,nofbits)

    a_z=0
    b_z=0
    if(a_y!=0):
      a_z = int(math.log2(a_y))
    if(b_y!=0):
      b_z = int(math.log2(b_y))

    multin_a = multiplixing(a_y,a_z,mult_wd)
    multin_b = multiplixing(b_y,b_z,mult_wd)

    shamt_a = shamt(a_z,mult_wd)
    shamt_b = shamt(b_z,mult_wd)

    multin_a = int(multin_a) >> shamt_a
    multin_b = int(multin_b) >> shamt_b

    accurate_product = multin_a * multin_b
    t_shamt = shamt_a + shamt_b
    unsigned_result = accurate_product << t_shamt

    if (sign_a ^ sign_b):
        return -1*unsigned_result
    return unsigned_result

def conv2d(inp, weights, bias, nofbits, mult_wd):
  if(len(inp.shape)==2):
    return conv2d_2d(inp, weights, bias, nofbits, mult_wd)
  return conv2d_3d(inp, weights, bias, nofbits, mult_wd)

def conv2d_2d(inp, weights, bias, nofbits, mult_wd):
  (inp_w, inp_h) = inp.shape
  weights = np.squeeze(weights)
  (filter_w, filter_h, num_filters) = weights.shape
  out_matrix = np.zeros((inp_w-filter_w+1, inp_h-filter_h+1, num_filters))
  for curr_filter in range(num_filters):
    for x in range(inp_w-filter_w+1):
      for y in range(inp_h-filter_h+1):
        mult_acc = 0
        for filter_x in range(filter_w):
          for filter_y in range(filter_h):
            inputValue = inp[x+filter_x][y+filter_y]
            kernalValue = weights[filter_x][filter_y][curr_filter]
            newVal = multiplyNumbers(inputValue, kernalValue, nofbits, mult_wd)
            mult_acc += newVal
        out_matrix[x][y][curr_filter] = mult_acc + bias[curr_filter]
  return relu(out_matrix)

def conv2d_3d(inp, weights, bias, nofbits, mult_wd):
  (inp_l, inp_w, inp_h) = inp.shape
  weights = np.squeeze(weights)
  (filter_l, filter_w, filter_h, num_filters) = weights.shape
  out_matrix = np.zeros((inp_l-filter_l+1, inp_w-filter_w+1, num_filters))
  for curr_filter in range(num_filters):
    for z in range(inp_l-filter_l+1):
      for x in range(inp_w-filter_w+1):
        for y in range(inp_h-filter_h+1):
          mult_acc = 0
          for filter_z in range(filter_l):
            for filter_x in range(filter_w):
              for filter_y in range(filter_h):
                inputValue = inp[z+filter_z][x+filter_x][y+filter_y]
                kernalValue = weights[filter_z][filter_x][filter_y][curr_filter]
                newVal = multiplyNumbers(inputValue, kernalValue, nofbits, mult_wd)
                mult_acc += newVal
          out_matrix[z][x][curr_filter] = mult_acc + bias[curr_filter]
  return relu(out_matrix)

# a simple dense layer
def dense(inp, weights, bias, nofbits, mult_wd):
  (inp_len, out_len) = weights.shape
  out_matrix = np.zeros((out_len))
  for i in range(out_len):
    mult_acc = 0
    for j in range(inp_len):
      mult_acc+=multiplyNumbers(inp[j],weights[j][i], nofbits, mult_wd)
    out_matrix[i] = mult_acc + bias[i]
  return out_matrix + bias

# relu activation
def relu(data):
  return np.maximum(data, 0)

def to_int(data, quantization_precision = 100):
    decimals = int(math.log10(quantization_precision))
    return (np.round(data, decimals=decimals)*quantization_precision).astype(int)

def getBareBonesModel(model, quantization_precision):
    barebones_model = []
    for layer in model.layers:
        try:
            if(layer.name=="flatten"):
                barebones_model.append(("flatten", [], []))
            elif(layer.name[:len("dense")]=="dense" or layer.name[:len("conv2d")]=="conv2d"):
                weights = to_int(layer.weights[0].numpy(), quantization_precision)
                bias = to_int(layer.weights[1].numpy(), quantization_precision)
                if(layer.name[:len("conv2d")]=="conv2d"):
                    barebones_model.append(("conv2d", weights, bias))
                elif(layer.name[:len("dense")]=="dense"):
                    barebones_model.append(("dense", weights, bias))
            elif(layer.name[:len("reshape")]=="reshape"):
                continue
            else:
                raise LayerNotSupported
        except LayerNotSupported:
                print("*"*60)
                print("[WARNING] ApproximateNeuralNetwork.py")
                print("Layer '" + str(layer.name) +  "' is not a supported layer type")
                print("If your model does not work, this is likely the reason")
                print("*"*60)
    return barebones_model