import torch
import torchvision
import numpy as np
import os
import csv
import argparse
from datetime import datetime
from quantize_neural_net import QuantizeNeuralNet
from utils import test_accuracy, eval_sparsity, fusion_layers_inplace
from data_loaders import data_loader

LOG_FILE_NAME = '../logs/Quantization_Log.csv'

# hyperparameter section
parser = argparse.ArgumentParser(description='Stochastic Quantization')

parser.add_argument('--bits', '-b', default=[4], type=int, nargs='+',
                    help='number of bits for quantization')
parser.add_argument('--scalar', '-s', default=[1.16], type=float, nargs='+',
                    help='the scalar C used to determine the radius of alphabets')
parser.add_argument('--batch_size', '-bs', default=[128], type=int, nargs='+',
                    help='batch size used for quantization')
parser.add_argument('--percentile', '-p', default=[1], type=float, nargs='+',
                    help='percentile of weights')
parser.add_argument('--num_worker', '-w', default=8, type=int, 
                    help='number of workers for data loader')
parser.add_argument('--data_set', '-ds', default='ILSVRC2012', choices=['ILSVRC2012', 'CIFAR10'],
                    help='dataset used for quantization')
parser.add_argument('-model', default='resnet18', help='model name')   
parser.add_argument('--stochastic_quantization', '-sq', action='store_true',
                    help='use stochastic quantization')
parser.add_argument('--retain_rate', '-rr', default=0.25, type=float,
                    help='subsampling probability p for convolutional layers')
parser.add_argument('--regularizer', '-reg', default=None, choices=['L0', 'L1'], 
                    help='choose the regularization mode')
parser.add_argument('--lamb', '-l', default=[0.1], type=float, nargs='+',
                    help='regularization term')
parser.add_argument('--ignore_layer', '-ig', default=[], type=int, nargs='+',
                    help='indices of unquantized layers')
parser.add_argument('-seed', default=0, type=int, help='set random seed')
parser.add_argument('--fusion', '-f', action='store_true', help='fusing CNN and BN layers')

args = parser.parse_args()


def main(b, mlp_s, cnn_s, bs, mlp_per, cnn_per, l):
    batch_size = bs  
    bits = b
    mlp_percentile = mlp_per 
    cnn_percentile = cnn_per
    mlp_scalar = mlp_s 
    cnn_scalar = cnn_s
    lamb = l
    stochastic = args.stochastic_quantization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the model to be quantized
    if args.data_set == 'ILSVRC2012':
        model = getattr(torchvision.models, args.model)(pretrained=True)

        # NOTE: refer to https://pytorch.org/vision/stable/models.html
        original_accuracy_table = {
            'alexnet': (.56522, .79066),
            'vgg16': (.71592, .90382),
            'resnet18': (.69758, .89078),
            'googlenet': (.69778, .89530),
            'resnet50': (.7613, .92862),
            'efficientnet_b1': (.7761, .93596),
            'efficientnet_b7': (.84122, .96908),
            'mobilenet_v2': (.71878, .90286)
        }

    elif args.data_set == 'CIFAR10':
        model = torch.load(os.path.join('pretrained_cifar10', args.model + '_cifar10.pt'), 
            map_location=torch.device('cpu')).module

        original_accuracy_table = {}
    
    model.to(device)  
    model.eval()  # turn on the evaluation mode

    if args.fusion:
        fusion_layers_inplace(model, device) # combine CNN layers and BN layers (in place fusion)
        print('CNN and BN layers are fused before quantization!\n')
    
    if stochastic:
        print(f'Quantization mode: stochastic quantization, i.e. SGPFQ')
    elif args.regularizer:
        print(f'Quantization mode: sparse quantization using {args.regularizer} norm with lambda {lamb}')
    else:
        print(f'Quantization mode: GPFQ')
        
    print('\nQuantization hyperparameters:')
    print(f'Quantizing {args.model} on {device} with\n\t  dataset: {args.data_set}, bits: {bits}, mlp_scalar: {mlp_scalar}, cnn_scalar: {cnn_scalar}, mlp_percentile: {mlp_percentile}, \
        \n\tcnn_percentile: {cnn_percentile}, retain_rate: {args.retain_rate}, batch_size: {batch_size}\n')
    
    
    # load the data loader for training and testing
    train_loader, test_loader = data_loader(args.data_set, batch_size, args.num_worker)
    
    # quantize the neural net
    quantizer = QuantizeNeuralNet(model, args.model, batch_size, 
                                    train_loader, 
                                    mlp_bits=bits,
                                    cnn_bits=bits,
                                    ignore_layers=args.ignore_layer,
                                    mlp_alphabet_scalar=mlp_scalar,
                                    cnn_alphabet_scalar=cnn_scalar,
                                    mlp_percentile=mlp_percentile,
                                    cnn_percentile=cnn_percentile,
                                    reg = args.regularizer, 
                                    lamb=lamb,
                                    retain_rate=args.retain_rate,
                                    stochastic_quantization=stochastic,
                                    device = device
                                    )
    start_time = datetime.now()
    quantized_model = quantizer.quantize_network()
    end_time = datetime.now()
    quantized_model = quantized_model.to(device)

    print(f'\nTime used for quantization: {end_time - start_time}\n')

    saved_model_name = f'ds{args.data_set}_b{bits}_batch{batch_size}_mlpscalar{mlp_scalar}_cnnscalar{cnn_scalar}\
        _mlppercentile{mlp_percentile}_cnnpercentile{cnn_percentile}_retain_rate{args.retain_rate}\
        _reg{args.regularizer}_lambda{lamb}.pt'

    if not os.path.isdir('../quantized_models/'):
        os.mkdir('../quantized_models/')
    saved_model_dir = '../quantized_models/'+args.model
    if not os.path.isdir(saved_model_dir):
        os.mkdir(saved_model_dir)
    torch.save(quantized_model, os.path.join(saved_model_dir, saved_model_name))

    topk = (1, 5)   # top-1 and top-5 accuracy
    
    if args.model in original_accuracy_table:
        print(f'\nUsing the original model accuracy from pytorch.\n')
        original_topk_accuracy = original_accuracy_table[args.model]
    else:
        print(f'\nEvaluting the original model to get its accuracy\n')
        original_topk_accuracy = test_accuracy(model, test_loader, device, topk)
    
    print(f'Top-1 accuracy of {args.model} is {original_topk_accuracy[0]}.')
    print(f'Top-5 accuracy of {args.model} is {original_topk_accuracy[1]}.')
    
    start_time = datetime.now()

    print(f'\n Evaluting the quantized model to get its accuracy\n')
    topk_accuracy = test_accuracy(quantized_model, test_loader, device, topk)
    print(f'Top-1 accuracy of quantized {args.model} is {topk_accuracy[0]}.')
    print(f'Top-5 accuracy of quantized {args.model} is {topk_accuracy[1]}.')

    end_time = datetime.now()

    print(f'\nTime used for evaluation: {end_time - start_time}\n')
    
    original_sparsity = eval_sparsity(model)
    quantized_sparsity = eval_sparsity(quantized_model)
    
    print("Sparsity: Org: {}, Quant: {}".format(original_sparsity, quantized_sparsity))
    # store the validation accuracy and parameter settings
    with open(LOG_FILE_NAME, 'a') as f:
        csv_writer = csv.writer(f)
        row = [
            args.model, args.data_set, batch_size, 
            original_topk_accuracy[0], topk_accuracy[0], 
            original_topk_accuracy[1], topk_accuracy[1], 
            bits, mlp_scalar, cnn_scalar, 
            mlp_percentile, cnn_percentile, stochastic,
            args.regularizer, lamb, original_sparsity, quantized_sparsity,
            args.retain_rate, args.fusion, args.seed
        ]
        csv_writer.writerow(row)


if __name__ == '__main__':

    params = [(b, s, s, bs, mlp_per, cnn_per, l) 
                            for b in args.bits
                            for s in args.scalar
                            for bs in args.batch_size
                            for mlp_per in args.percentile
                            for cnn_per in args.percentile
                            for l in args.lamb
                            ]

    # testing section
    for b, mlp_s, cnn_s, bs, mlp_per, cnn_per, l in params:
        main(b, mlp_s, cnn_s, bs, mlp_per, cnn_per, l)
