import csv

log_file_name = 'Quantization_Log.csv'

fields = ['Model Name', 'Dataset', 'Quantization Batch Size', 'Original Top1 Accuracy',
          'Quantized Top1 Accuracy', 'Original Top5 Accuracy', 'Quantized Top5 Accuracy', 'Bits', 'Alphabet_Scalar', 
          'Include 0', 'Seed', 'Author']

if __name__ == '__main__':
    with open(log_file_name, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()
