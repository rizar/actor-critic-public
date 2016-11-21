function bleu_dev_short {
     for f in `ls | sort -g`
     do 
         echo $f
         cat $f |  ~/Dist/mosesdecoder/scripts/generic/multi-bleu.perl\
             <(  awk '{if (NF <= 25) print}' $FUEL_DATA_PATH/TED/de-en/prep/dev.de-en.en )
     done
}

function just_bleu_dev_short {
    for f in `ls | sort -g`; 
    do 
        cat $f | ~/Dist/mosesdecoder/scripts/generic/multi-bleu.perl\
            <(  awk '{if (NF <= 25)  print}' $FUEL_DATA_PATH/TED/de-en/prep/dev.de-en.en  ) |
            awk '{print $3}'
    done
}

function bleu_train_short {
    for f in `ls | sort -g`; 
    do 
        echo $f 
        cat $f | ~/Dist/mosesdecoder/scripts/generic/multi-bleu.perl\
             <(  awk '{if (NF <= 25)  print}' $FUEL_DATA_PATH/TED/de-en/prep/train.de-en.en | head -3000 )
    done
}

function just_bleu_train_short {
    for f in `ls | sort -g`; 
    do 
        cat $f | ~/Dist/mosesdecoder/scripts/generic/multi-bleu.perl\
            <(  awk '{if (NF <= 25)  print}' $FUEL_DATA_PATH/TED/de-en/prep/train.de-en.en | head -3000 ) |
            awk '{print $3}'
    done
}

function bleu_dev_long {
    for f in `ls | sort -g`; 
    do 
        echo $f 
        cat $f | ~/Dist/mosesdecoder/scripts/generic/multi-bleu.perl $FUEL_DATA_PATH/TED/de-en/prep/dev.de-en.en
    done
}

function bleu_train_long {
    for f in `ls | sort -g`; 
    do 
        echo $f 
        cat $f | ~/Dist/mosesdecoder/scripts/generic/multi-bleu.perl <( head -3000 $FUEL_DATA_PATH/TED/de-en/prep/train.de-en.en )
    done
}
