import os

data_folder = '../data/en-fr/raw/'

train_src = os.path.join(data_folder, 'train.fr')
train_trg = os.path.join(data_folder, 'train.en')
tiny_train_src = os.path.join(data_folder, 'tiny_train.fr')
tiny_train_trg = os.path.join(data_folder, 'tiny_train.en')
valid_src = os.path.join(data_folder, 'valid.fr')
valid_trg = os.path.join(data_folder, 'valid.en')
test_src = os.path.join(data_folder, 'test.fr')
test_trg = os.path.join(data_folder, 'test.en')

src_codes = os.path.join(data_folder, 'src_codes.bpe')
tgt_codes = os.path.join(data_folder, 'tgt_codes.bpe')

train_src_bpe = os.path.join(data_folder, 'train_bpe.fr')
train_trg_bpe = os.path.join(data_folder, 'train_bpe.en')
tiny_train_src_bpe = os.path.join(data_folder, 'tiny_train_bpe.fr')
tiny_train_trg_bpe = os.path.join(data_folder, 'tiny_train_bpe.en')
valid_src_bpe = os.path.join(data_folder, 'valid_bpe.fr')
valid_trg_bpe = os.path.join(data_folder, 'valid_bpe.en')
test_src_bpe = os.path.join(data_folder, 'test_bpe.fr')
test_trg_bpe = os.path.join(data_folder, 'test_bpe.en')

# Train BPE models using the command line
os.system(f'subword-nmt learn-bpe -s 32000 < {train_src} > {src_codes}')
os.system(f'subword-nmt learn-bpe -s 32000 < {train_trg} > {tgt_codes}')

# Apply BPE to your data using the command line
os.system(f'subword-nmt apply-bpe -c {src_codes} < {train_src} > {train_src_bpe}')
os.system(f'subword-nmt apply-bpe -c {tgt_codes} < {train_trg} > {train_trg_bpe}')

os.system(f'subword-nmt apply-bpe -c {src_codes} < {tiny_train_src} > {tiny_train_src_bpe}')
os.system(f'subword-nmt apply-bpe -c {tgt_codes} < {tiny_train_trg} > {tiny_train_trg_bpe}')

# Repeat the above steps for valid and test data
os.system(f'subword-nmt apply-bpe -c {src_codes} < {valid_src} > {valid_src_bpe}')
os.system(f'subword-nmt apply-bpe -c {tgt_codes} < {valid_trg} > {valid_trg_bpe}')

os.system(f'subword-nmt apply-bpe -c {src_codes} < {test_src} > {test_src_bpe}')
os.system(f'subword-nmt apply-bpe -c {tgt_codes} < {test_trg} > {test_trg_bpe}')