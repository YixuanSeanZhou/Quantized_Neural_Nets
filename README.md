# Post-training Quantization for Neural Networks with Provable Guarantees

#### Authors: [Jinjie Zhang](https://scholar.google.com/citations?user=YCR4koUAAAAJ&hl=en) (jiz003@ucsd.edu), [Yixuan Zhou](https://yixuanseanzhou.github.io/) (yiz044@ucsd.edu) and [Rayan Saab](https://mathweb.ucsd.edu/~rsaab/) (rsaab@ucsd.edu)

## Overview 
This directory contains code necessary to run a post-training neural-network quantization method GPFQ, that
is based on a greedy path-following mechanism. One can also use it to reproduce the experiment results in our paper ["Post-training Quantization for Neural Networks with Provable Guarantees"](). In this paper, we also prove theoretical guarantees for the proposed method, that is, for quantizing a single-layer network, the relative square error essentially decays linearly in the number of weights – i.e., level of over-parametrization. 

If you make use of this code or our quantization method in your work, please cite the following paper:

     @inproceedings{,
	     author = {Zhang, Jinjie and Zhou, Yixuan and Saab, Rayan},
	     title = {Post-training Quantization for Neural Networks with Provable Guarantees},
	     booktitle = {},
	     year = {2022}
	   }


*Note:* The project mainly consider the ImageNet dataset, and due to the size of this dataset we strongly recommend one to run this experiment using a cloud computation center, e.g. AWS. When we run this experiment, we use the `m5.8xlarge` AWS EC2 instance with a disk space of `300GB`.

## Installing Dependencies
We assume a python version that is greater than `3.8.0` is installed in the user's 
machine. In the root directory of this repo, we provide a `requirements.txt` file for installing the python libraries that will be used in our code. 

To install the necessary dependency, one can first start a virtual environment
by doing the following: 
```
python3 -m venv .venv
source .venv/bin/activate
```
The code above should activate a new python virtual environments.

Then one can make use of the `requirements.txt` by 
```
pip3 install -r requirement.txt
```
This should install all the required dependencies of this project. 

## Obtaining ImageNet Dataset

In this project, we make use of the Imagenet dataset, 
in particular, we use the ILSVRC-2012 version. 

To obtain the Imagenet dataset, one can submit a request through this [link](https://image-net.org/request).

Once the dataset is obtained, place the `.tar` files for training set and validation set both under the `data/ILSVRC2012` directory of this repo. 

Then use the following procedure to unzip Imagenet dataset:
```
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
# Extract the validation data and move images to subfolders:
tar -xvf ILSVRC2012_img_val.tar
``` 

## Running Experiments

The implementation of the modified GPFQ in our paper is contained in `quantization_scripts`. Additionally, `adhoc_quantization_scripts` and `retraining_scripts` provide extra experiments and both of them are variants of the framework in `quantization_scripts`. `adhoc_quantization_scripts` contains heuristic modifications used to further improve the performance of GPFQ, such as bias correction, mixed precision, and unquantizing the last layer. `retraining_scripts` shows a quantization-aware training strategy that is designed to retrain the neural network after each layer is quantized. 

In this section, we will give a guidance on running our code contained in `quantization_scripts` and the implementation of other two counterparts `adhoc_quantization_scripts` and `retraining_scripts` are very similar to `quantization_scripts`.

1. Before getting started, run in the root directory of the repo and run `mkdir models`to create a directory in which we will store the quantized model. 

2. The entry point of the project starts with `quantization_scripts/quantize.py`. 
Once the file is opened, there is a section to set hyperparameters, for example, the `model_name` parameter, the number of bits/batch size used for quantization, the scalar of alphabets, the probability for subsampling in CNNs etc. Note that the `model_name` mentioned above should be the same as the model that you will quantize. After you selected a `model_name` and assuming you are still in the root directory of this repo, run `mkdir models/{model_name}`, where the `{model_name}` should be the python string that you provided for the `model_name` parameter in the `quantize.py` file. If the directory already exists, you can skip this step. 

3. Then navigate to the `logs` directory and run `python3 init_logs.py`. This will prepare a log file which is used to store the results of the experiment.

4. Finally, open the `quantization_scripts` directory and run `python3 quantize.py` to start the experiment.
