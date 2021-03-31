# Approximate Multiply Quantized Neural Networks
This project implements approximate multiplication in neural networks based on [DRUM: A Dynamic Range Unbiased Multiplier for Approximate Applications](https://www.researchgate.net/publication/304252296_DRUM_A_Dynamic_Range_Unbiased_Multiplier_for_Approximate_Applications)

## Getting started
Use exampleModel.py as a reference for implementing your own networks

## AppromixateNeuralNetwork.py
Usage:
```
ApproximateNeuralNetwork(model)
```
Where model is a standard TensorFlow model

Variables:
```
quantization_precision # default 100, constant to multiply weights by to turn them into integers. Must be a power of 10
mult_wd # default 3, number of bits to perform accurate multiply on
nofbits # default 8, number of bits for weights, bias, and activations
```

## Supported Layer Types:
* dense
* max_pooling2d
* avg_pooling2d
* conv2d (only with relu activation, and strides of (1,1))
* flatten
* reshape (is simply ignored)
