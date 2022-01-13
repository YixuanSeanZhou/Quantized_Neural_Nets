# Quantized_Neural_Nets

## Oct. 21th, 2021

### Yixuan's Update

Set up the pipeline of training `MLP`. 

TODO: Currently the accuracy depends highly on radius, need figure out how to deal with it.

## Oct. 17th, 2021

### Yixuan's Update

Set up a general framework of quantizing neural networks.
The idea is we should have a runner that runs instantiate `QuantizeNeuralNet` 
that is defined in `quantize_neural_net.py` to start the quantization process.
After that, it should call `quantize_network` on the `QuantizeNeuralNet` 
instantiated to perform the quantization, which return a reference of the 
network that is quantized.


### Jinjie's Update 

To unzip Imagenet dataset, run the following procedure: 

First, move the `.tar` file of the Imagenet dataset into the current working directory. Then run the following commands:

```
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
# Extract the validation data and move images to subfolders:
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
``` 
