import os

data_folder = '../data/en-fr/raw/'


# Train BPE models using the command line
L1 = 'fr'
L2 = 'en'
train_file = os.path.join(data_folder, 'train')
codes_file = os.path.join(data_folder, 'bpe_code')
vocab_file = os.path.join(data_folder, 'dict')

num_operations = 32000

os.system(f'subword-nmt learn-joint-bpe-and-vocab --input {train_file}.{L1} {train_file}.{L2} -s {num_operations} -o {codes_file} --write-vocabulary {vocab_file}.{L1} {vocab_file}.{L2}')

# Apply BPE to your data using the command line
for split in ['train', 'tiny_train', 'valid', 'test']:
    split_file = os.path.join(data_folder, split)
    for L in [L1, L2]:
        os.system(f'subword-nmt apply-bpe -c {codes_file} --vocabulary {vocab_file}.{L} --vocabulary-threshold 1 < {split_file}.{L} > {split_file}.BPE.{L}')
