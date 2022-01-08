import csv
from datetime import datetime
import os

bias_log_file_name = 'Quantization_Log_Bias_Correct.csv'

fields = ['Model Name', 'Dataset', 'Quantization Batch Size', 'Bias Correction Batch Size', 'Original Top1 Accuracy',
          'Quantized Top1 Accuracy', 'Original Top5 Accuracy', 'Quantized Top5 Accuracy', 
          'Bits', 'MLP_Alphabet_Scalar', 'CNN_Alphabet_Scalar', 'MLP_Percentile', 'CNN_Percentile',
          'Include 0', 'Retain_rate', 'Seed', 'Author']

if __name__ == '__main__':
    if os.path.isfile(bias_log_file_name):
        os.rename(bias_log_file_name, f'{datetime.now()}_{bias_log_file_name}')
    with open(bias_log_file_name, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
    
