Machine Translation experiments on TED dataset
==============================================

This folder contains scripts and configurations for 
experiments on TED dataset. 

To download and preprocess dataset, run ``create_dataset.sh``. If you are in MILA
you can skip this and use already preprocessed data from ``/data/lisatmp4/bahdanau/data``
(just set ``FUEL_DATA_PATH`` to point to this directory).

To reproduce the results from the paper, following the procedure below.

1. Train a model with the configuration ``ted12.yaml``. This should give you
a model pretrained with maximum log-likelihood (``main_best.tar``), and also
an annealed version of it (``annealing_best.tar``). The exact command is ``$LVSR/bin/run.py train ted12 ted12.yaml``.

2. Start actor-critic training using ``ted16.yaml`` and the additional options
``--start-stage critic_pretraining --params ted12/main_best.tar``. Wait until the training transitions to the main stage, the final outcome is ``ted16/main_best.tar``.

3. Use the configuration ``ted17.yaml`` to reproduce REINFORCE-critic results. 
You can start training from the main stage with parameters ``ted16/critic_pretraining.tar``. The deliverable is ``main_best.tar``.

4. For REINFORCE with linear baseline, run ``reinforced3.yaml`` starting 
from ``ted12/main_best.tar``.

We also recommend to use the script ``decode.sh`` for decoding. Please consult
the appendix of the paper for the exact value of the character discount.

For computing the BLEU score run 

```$LVSR/bin/extract_recognized.sh <$OUTPUT | $MOSES/scripts/generic/multi-bleu.perl $FUEL_DATA_PATH/TED/de-en/prep/dev.de-en.en```

where ``$MOSES`` is the path to Moses, ``$OUTPUT`` is an output produced by ``$LVSR/bin/run.py search``
