#!/usr/bin/env bash

function run {
    if [ `hostname -d` = helios ]
    then            
        smart-dispatch -q gpu_1  -t $TIME launch bash -x $LVSR/bin/run_cluster.sh train $@
    else
        $LVSR/bin/run.py train $@
    fi        
}

# Actor pretraining

if [ 0 == 1 ];
then
    [ -a ted10 ] || TIME=12:00:00 run ted10 $LVSR/exp/ted/configs/ted10.yaml
fi    

# Preliminary stuff

if [ 0 == 1 ];
then    
    [ -a ted2 ] || run --params ted1e/main_best.tar ted2 $LVSR/exp/ted/configs/ted2.yaml
    [ -a ted3 ] || run --params ted1e/main_best.tar ted3 $LVSR/exp/ted/configs/ted3.yaml
    [ -a ted3k ] || TIME=23:59:59 run --params ted1e/main_best.tar ted3k $LVSR/exp/ted/configs/ted3.yaml stages.critic_pretraining.net.criterion.softmax_t 2.
    [ -a ted5 ] || TIME=12:00:00 run ted5 $LVSR/exp/ted/configs/ted5.yaml
fi    

# Edit distance

if [ 0 == 1 ];
then      
    [ -a ted6 ] || TIME=23:59:59 run --params ted1e/main_best.tar ted6 $LVSR/exp/ted/configs/ted6.yaml
    [ -a ted6c ] || TIME=23:59:59 run --params ted1e/main_best.tar ted6c $LVSR/exp/ted/configs/ted6.yaml training.scale 0.002
    [ -a ted6d ] || TIME=23:59:59 run --params ted1e/main_best.tar ted6d $LVSR/exp/ted/configs/ted6.yaml training.momentum 0.5
    [ -a ted6e ] || TIME=23:59:59 run --params ted1e/main_best.tar ted6e $LVSR/exp/ted/configs/ted6.yaml training.momentum 0.98
    [ -a ted6f ] || TIME=23:59:59 run --params ted1e/main_best.tar ted6f $LVSR/exp/ted/configs/ted6.yaml training.external_targets False training.external_policy True
    [ -a ted6g ] || TIME=23:59:59 run --params ted1e/main_best.tar ted6g $LVSR/exp/ted/configs/ted6.yaml training.gradient_threshold 10.
    [ -a ted6h ] || TIME=23:59:59 run --params ted1e/main_best.tar ted6h $LVSR/exp/ted/configs/ted6.yaml net.criterion.critic_uses_actor_states False
    [ -a ted6k ] || TIME=23:59:59 run --params ted1e/main_best.tar ted6k $LVSR/exp/ted/configs/ted6.yaml stages.critic_pretraining.net.criterion.softmax_t 2.
fi

# BLEU

if [ 0 == 1 ];
then
    [ -a ted4b ] || TIME=23:59:59 run --params ted1e/main_best.tar ted4b $LVSR/exp/ted/configs/ted4.yaml
    [ -a ted4c ] || TIME=23:59:59 run --params ted1e/main_best.tar ted4c $LVSR/exp/ted/configs/ted4.yaml training.scale 0.002
    [ -a ted4d ] || TIME=23:59:59 run --params ted1e/main_best.tar ted4d $LVSR/exp/ted/configs/ted4.yaml training.momentum 0.5
    [ -a ted4e ] || TIME=23:59:59 run --params ted1e/main_best.tar ted4e $LVSR/exp/ted/configs/ted4.yaml training.momentum 0.98
    [ -a ted4f ] || TIME=23:59:59 run --params ted1e/main_best.tar ted4f $LVSR/exp/ted/configs/ted4.yaml training.external_targets False training.external_policy True
    [ -a ted4g ] || TIME=23:59:59 run --params ted1e/main_best.tar ted4g $LVSR/exp/ted/configs/ted4.yaml training.gradient_threshold 10.
    [ -a ted4h ] || TIME=23:59:59 run --params ted1e/main_best.tar ted4h $LVSR/exp/ted/configs/ted4.yaml net.criterion.critic_uses_actor_states False
    [ -a ted4k ] || TIME=23:59:59 run --params ted1e/main_best.tar ted4k $LVSR/exp/ted/configs/ted4.yaml stages.critic_pretraining.net.criterion.softmax_t 2.
fi

# BLEU, all sentece without clipping
if [ 0 == 1 ];
then
    # WITH BUGS
    # baseline: 20.23
    # ted9: 19.47
    # ted9k: 18.61
    # $LVSR/bin/run.py train --params $TMP4/mt2/ted9/critic_pretraining.tar  --start-stage main ted91 $LVSR/exp/ted/configs/ted9.yaml net.criterion.entropy_reward_coof 0.0 
    # best: 20.15 after 33544
    # $LVSR/bin/run.py train --params $TMP4/mt2/ted9/critic_pretraining.tar  --start-stage main ted92 $LVSR/exp/ted/configs/ted9.yaml net.criterion.entropy_reward_coof 0.0 net.criterion.epsilon 0.0
    # best: 20.12 after 23960
    # WITHOUT BUGS
    # $LVSR/bin/run.py train --params $TMP4/mt2/ted1i/main_best.tar  ted9l $LVSR/exp/ted/configs/ted9.yaml net.criterion.softmax_t 2.
    # best: 20.63 after 9584
    # $LVSR/bin/run.py train --params $TMP4/mt2/ted1i/main_best.tar  ted9c $LVSR/exp/ted/configs/ted9.yaml
    # best: 20.73 after 14376
    echo
fi

# BLEU <= 25
if [ 1 == 1 ];
then
    # baseline: 18.96
    [ -a ted11 ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11 $LVSR/exp/ted/configs/ted11.yaml
    # best: 19.49
    # $LVSR/bin/run.py train --params $TMP4/mt2/ted11/critic_pretraining.tar  --start-stage main ted111 $LVSR/exp/ted/configs/ted11.yaml net.criterion.entropy_reward_coof 0.0 net.criterion.epsilon 0.0 
    # best: 19.88
    [ -a ted11a ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11a $LVSR/exp/ted/configs/ted11.yaml net.criterion.epsilon 0.0
    # best: 18.70 (but starts from as low as 16.04)
    [ -a ted11b ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11b $LVSR/exp/ted/configs/ted11.yaml net.criterion.entropy_reward_coof 0.0
    # best: 20.04 after 18410 updates
    [ -a ted11c ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11c $LVSR/exp/ted/configs/ted11.yaml net.criterion.reward bleu
    # best: did not work
    [ -a ted11d ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11d $LVSR/exp/ted/configs/ted11.yaml net.criterion.discount 0.9
    # best: 18.66
    [ -a ted11e ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11e $LVSR/exp/ted/configs/ted11.yaml stages.critic_pretraining.net.criterion.softmax_t 2.
    # best: 20.03
    [ -a ted11f ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11f $LVSR/exp/ted/configs/ted11.yaml training.catching_up_coof 0.01
    # best: 18.90
    [ -a ted11g ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11g $LVSR/exp/ted/configs/ted11.yaml training.catching_up_coof 0.1
    # best: 17.07
    [ -a ted11h ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11h $LVSR/exp/ted/configs/ted11.yaml net.criterion.solve_bellman "'without_dp'"
    # best: 13.68
    # $LVSR/bin/run.py train --params $TMP4/mt2/ted11/critic_pretraining.tar  --start-stage main ted111 $LVSR/exp/ted/configs/ted11.yaml net.criterion.entropy_reward_coof 0.0 net.criterion.epsilon 0.0
    # best: 19.88 after 29456 updates
    [ -a ted11i ] || (mkdir ted11i; TIME=12:00:00 run --start-stage main --params ted11b/critic_pretraining.tar ted11i $LVSR/exp/ted/configs/ted11.yaml net.criterion.epsilon 0.0 net.criterion.entropy_reward_coof 0.0)
    # best: 20.15 after 33138 updates
    [ -a ted11j ] || (mkdir ted11j; TIME=12:00:00 run --start-stage main --params ted11a/critic_pretraining.tar ted11j $LVSR/exp/ted/configs/ted11.yaml net.criterion.epsilon 0.0 net.criterion.entropy_reward_coof 0.0)
    # best: 19.48 after 22092 updates
    [ -a ted11k ] || (mkdir ted11k; TIME=12:00:00 run --start-stage main --params ted11e/critic_pretraining.tar ted11k $LVSR/exp/ted/configs/ted11.yaml net.criterion.epsilon 0.0 net.criterion.entropy_reward_coof 0.0)
    # best: 20.18 after 33138 updates
    [ -a ted11l ] || (mkdir ted11l; TIME=12:00:00 run --start-stage main --params ted11e/critic_pretraining.tar ted11l $LVSR/exp/ted/configs/ted11.yaml net.criterion.epsilon 0.0 net.criterion.entropy_reward_coof 0.0 net.criterion.softmax_t 2.)
    # best: 20.73 after 22092 updates
    [ -a ted11n ] || (mkdir ted11n; TIME=12:00:00 run --start-stage main --params ted11b/critic_pretraining.tar ted11n $LVSR/exp/ted/configs/ted11.yaml net.criterion.epsilon 0.0 net.criterion.entropy_reward_coof 0.0 training.catching_up_coof 0.0001) 
    # best: 20.35 after 33138 updates
    [ -a ted11m ] || (mkdir ted11m; TIME=12:00:00 run --start-stage main --params ted11b/critic_pretraining.tar ted11m $LVSR/exp/ted/configs/ted11.yaml net.criterion.epsilon 0.0 net.criterion.entropy_reward_coof 0.0 training.catching_up_coof 1 training.catching_up_freq 5000) 
    # best: 19.16 after 25774 updates
    [ -a ted11o ] || (mkdir ted11o; TIME=12:00:00 run --start-stage main --params ted11b/critic_pretraining.tar ted11o $LVSR/exp/ted/configs/ted11.yaml net.criterion.epsilon 0.0 net.criterion.entropy_reward_coof 0.1)
    # best: 19.49 after 22092 updates

    # AFTER RUSH
    [ -a ted11q ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11q\
        $LVSR/exp/ted/configs/ted11.yaml\
        stages.critic_pretraining.net.criterion.epsilon 0.0\
        net.criterion.same_value_for_wrong False
    [ -a ted11r ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11r\
        $LVSR/exp/ted/configs/ted11.yaml\
        stages.critic_pretraining.net.criterion.epsilon 0.0\
        net.criterion.same_value_for_wrong False\
        net.criterion.critic_uses_actor_states True
    [ -a ted11s ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11s\
        $LVSR/exp/ted/configs/ted11.yaml\
        net.criterion.same_value_for_wrong False
    [ -a ted11t ] || TIME=23:59:59 run --params ted10/main_best_ll.tar ted11t\
        $LVSR/exp/ted/configs/ted11.yaml\
        net.criterion.same_value_for_wrong False\
        net.criterion.critic_uses_actor_states True
    # /u/bahdanau/Dist/fully-neural-lvsr/bin/run.py train --params 
    # $TMP4/mt2/ted10/main_best_ll.tar   ted11u
    # /u/bahdanau/Dist/fully-neural-lvsr/exp/ted/configs/ted11.yaml
    # stages.critic_pretraining.net.criterion.epsilon 0.0
    # net.criterion.same_value_for_wrong False net.criterion.value_penalty 0.001 
    # BEST: 19.47 after 22092
    # /u/bahdanau/Dist/fully-neural-lvsr/bin/run.py train --params
    # $TMP4/mt2/ted11u/critic_verbose.tar --start-stage main ted11u1
    # /u/bahdanau/Dist/fully-neural-lvsr/exp/ted/configs/ted11.yaml
    # stages.critic_pretraining.net.criterion.epsilon 0.0
    #  net.criterion.same_value_for_wrong False net.criterion.value_penalty 0.001
    # training.gradient_threshold 10.0
    # BEST: 19.99 after 22092 
    # /u/bahdanau/Dist/fully-neural-lvsr/bin/run.py train --save-inter
    # --params $TMP4/mt2/ted11u/critic_verbose.tar --start-stage main ted11u4
    # /u/bahdanau/Dist/fully-neural-lvsr/exp/ted/configs/ted11.yaml
    # stages.critic_pretraining.net.criterion.epsilon 0.0
    # net.criterion.same_value_for_wrong False net.criterion.value_penalty 0.001
    # training.gradient_threshold 10.0 net.criterion.average_log_coof 0.000001
    # BEST: diverged
    # /u/bahdanau/Dist/fully-neural-lvsr/bin/run.py train --save-inter --params
    # $TMP4/mt2/ted11u/critic_verbose.tar --start-stage main ted11u5
    # /u/bahdanau/Dist/fully-neural-lvsr/exp/ted/configs/ted11.yaml
    # stages.critic_pretraining.net.criterion.epsilon 0.0
    # net.criterion.same_value_for_wrong False net.criterion.value_penalty 0.001
    # training.gradient_threshold 10.0 net.criterion.average_log_coof 0.0000001
    # BEST: 20.09
    # /u/bahdanau/Dist/fully-neural-lvsr/bin/run.py train --params
    # /data/lisatmp4/bahdanau/mt2/ted11u/critic_verbose.tar   --start-stage  main
    # ted11u6 /u/bahdanau/Dist/fully-neural-lvsr/exp/ted/configs/ted11.yaml
    # net.criterion.same_value_for_wrong False net.criterion.value_penalty 0.001
    # training.gradient_threshold 10.0 net.criterion.average_log_coof 0.0000001
    # monitoring.monitor_parameters True training.catching_up_coof 0.0001
    # BEST: 20.21
    # /u/bahdanau/Dist/fully-neural-lvsr/bin/run.py train --params
    # /data/lisatmp4/bahdanau/mt2/ted11u/critic_verbose.tar   --start-stage  main
    # ted11u7 /u/bahdanau/Dist/fully-neural-lvsr/exp/ted/configs/ted11.yaml
    # net.criterion.same_value_for_wrong False net.criterion.value_penalty 0.001
    # training.gradient_threshold 10.0 training.scale 0.0001
    # BEST: 21.35
    # /u/bahdanau/Dist/fully-neural-lvsr/bin/run.py train --params
    # /data/lisatmp4/bahdanau/mt2/ted11u/critic_verbose.tar   --start-stage
    # main ted11u8 /u/bahdanau/Dist/fully-neural-lvsr/exp/ted/configs/ted11.yaml
    # net.criterion.same_value_for_wrong False net.criterion.value_penalty 0.001
    # training.gradient_threshold 10.0 training.catching_up_coof 0.0001
    # BEST: 20.15
fi
