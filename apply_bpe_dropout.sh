#!/bin/bash
# -*- coding: utf-8 -*-

set -e

# specify respective data folders
pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../..
train_pref=$1 # train # for testing, use tiny_train
src=fr
tgt=en
dropout_rate=0.1

data=$pwd/data_bpe_do_$tgt-$src/

# take raw data, normalize, tokenize, truecase, then apply BPE with a dropout of 0.1
for lang in $src $tgt
do
    echo "apply bpe dropout at $dropout_rate on $data/$train_pref.$lang"
    cat $data/raw/$train_pref.$lang | perl $base/moses_scripts/normalize-punctuation.perl -l $lang | perl $base/moses_scripts/tokenizer.perl -l $lang -a -q | perl $base/moses_scripts/truecase.perl --model $data/preprocessed/tm.$lang | subword-nmt apply-bpe -c $data/preprocessed/bpe.codes --dropout $dropout_rate > $data/preprocessed/$train_pref.$lang
done

# create the prepared data (using the original vocabulary files!)
python $base/preprocess.py \
    --source-lang $src --vocab-src $data/prepared/dict.$src \
    --target-lang $tgt --vocab-trg $data/prepared/dict.$tgt \
    --train-prefix $data/preprocessed/$train_pref \
    --dest-dir $data/prepared

echo "$data/prepared updated!"