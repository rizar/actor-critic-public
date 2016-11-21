#!/usr/bin/env bash

for DISCOUNT in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do    
    $LVSR/bin/run.py search ../main_best.tar  $LVSR/exp/ted/configs/ted1.yaml monitoring.search.char_discount $DISCOUNT monitoring.search.beam_size 10 >dev.bs10.cd$DISCOUNT
done

