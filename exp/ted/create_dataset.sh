#!/usr/bin/env bash

# This was mostly copy-pasted from FACEBOOK script

SCRIPTS=$MOSES/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

URL="http://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz"
GZ=de-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=prep
tmp=prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 50
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $prep/dev.de-en.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $prep/train.de-en.$l

    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        > $prep/test.de-en.$l
done

# End of the copy-paste from FACEBOOK script

# Shuffle the dataset in order to use the first N thousand for measuring
# performance on the training set

paste -d '\t' $prep/train.de-en.en prep/train.de-en.de  | shuf >$tmp/shuffled_bitext
awk -F '\t' '{print $1}' $prep/tmp/shuffled_bitext >$prep/train.de-en.en
awk -F '\t' '{print $2}' $prep/tmp/shuffled_bitext >$prep/train.de-en.de


$LVSR/exp/ted/binarize.py -v 3 -p -e -o -d vocab.de prep/train.de-en.de prep/dev.de-en.de prep/test.de-en.de  
$LVSR/exp/ted/binarize.py -v 3 -p -e -o -d vocab.en prep/train.de-en.en prep/dev.de-en.en prep/test.de-en.en

rm ted.h5

$LVSR/bin/pack_to_hdf5.py ted.h5\
    -s train.de-en.de.pkl dev.de-en.de.pkl test.de-en.de.pkl\
    -t train.de-en.en.pkl dev.de-en.en.pkl test.de-en.en.pkl\
    -n train dev test

$LVSR/exp/ted/add_vocabulary.py ted.h5 sources vocab.de
$LVSR/exp/ted/add_vocabulary.py ted.h5 targets vocab.en
