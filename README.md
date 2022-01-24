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


## Install Dependencies
We assume a python version that is greater than `3.8.0` is installed in the user's 
machine.

In the root directory of this repo, we provide a `requirememts.txt` file for the
ease of installation.

To install the necessary dependency, one can first start a virtual environment
by doing the following: 
```
python3 -m venv .venv
source .venv/bin/activate
```
The above should activate a new python virtual environments.

Then, one can make use of the `requirements.txt` by 
```
pip3 install -r requirement.txt
```
This should install all the required dependencies of this project. 

### Obtaining Dataset.

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

Before getting started, run in the root directory of the repo, run `mkdir models`to create a directory in which we will store the quantized model. 

The entry point of the project starts with `quantization_scripts/quantize.py`. 
Once the file is opened, there is a hyperparameter section for one to specify the 
batch size that is used for quantization, the scalar of each type of the layers,
the percentile for each type of layers, and etc. One can specify the hyperparameters that will be used in this quantization experiment. 

One thing to note is the  `model_name` parameter. This `model_name` should be the same as the model that you try to quantize. After you selected a `model_name`, assuming you are still in the root directory of this repo, run `mkdir models/{model_name}`, where the `{model_name}` should be the string value that you provided for the `model_name` parameter in the `quantize.py` file. If the directory already exists, you can skip this step. 

Then navigate to the `logs` directory and run `python3 init_logs.py`. This will prepare a log file which is used to store the results of the experiment.

Lastly, navigate into the `quantization_scripts` directory, run `python3 quantize.py` to start the experiment.
