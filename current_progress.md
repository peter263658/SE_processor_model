# Current Doing
1. run kernel_size=(self.kernel_size, 1) instead of kernel_size=(self.kernel_size, 2)
2. change the fft from 256 to 128 apply to sep conv, but first check the result of the decrease kernel size 

# TODO
1. check the quantization  
2. check the sparsity of the weights


# current training step
1. fft = 256 with kernel = 1 (causal attention)
2. fft = 128 with kernel = 1 (causal attention)