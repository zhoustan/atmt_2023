#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

# change into base directory to ensure paths are valid
cd "$base"

# create preprocessed directory
mkdir -p "$data/preprocessed/"

# normalize and tokenize raw data
cat "$data/raw/train.BPE.$src" | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > "$data/preprocessed/train.BPE.$src.p"
cat "$data/raw/train.BPE.$tgt" | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > "$data/preprocessed/train.BPE.$tgt.p"

# train truecase models
perl moses_scripts/train-truecaser.perl --model "$data/preprocessed/tm.$src" --corpus "$data/preprocessed/train.BPE.$src.p"
perl moses_scripts/train-truecaser.perl --model "$data/preprocessed/tm.$tgt" --corpus "$data/preprocessed/train.BPE.$tgt.p"

# apply truecase models to splits
cat "$data/preprocessed/train.BPE.$src.p" | perl moses_scripts/truecase.perl --model "$data/preprocessed/tm.$src" > "$data/preprocessed/train.BPE.$src"
cat "$data/preprocessed/train.BPE.$tgt.p" | perl moses_scripts/truecase.perl --model "$data/preprocessed/tm.$tgt" > "$data/preprocessed/train.BPE.$tgt"

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat "$data/raw/$split.BPE.$src" | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model "$data/preprocessed/tm.$src" > "$data/preprocessed/$split.BPE.$src"
    cat "$data/raw/$split.BPE.$tgt" | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model "$data/preprocessed/tm.$tgt" > "$data/preprocessed/$split.BPE.$tgt"
done

# remove tmp files
rm "$data/preprocessed/train.BPE.$src.p"
rm "$data/preprocessed/train.BPE.$tgt.p"

# preprocess all files for model training
python preprocess_bpe.py --target-lang $tgt --source-lang $src --vocab-src "$data/raw/dict.$src" --vocab-trg "$data/raw/dict.$tgt" --dest-dir "$data/prepared/" --train-prefix "$data/preprocessed/train" --valid-prefix "$data/preprocessed/valid" --test-prefix "$data/preprocessed/test" --tiny-train-prefix "$data/preprocessed/tiny_train" --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000

cp "$data/raw/dict.$src" "$data/prepared/dict.$src"
cp "$data/raw/dict.$tgt" "$data/prepared/dict.$tgt"

echo "done!"