#!/usr/bin/env bash

function run {
    smart-dispatch -q gpu_1 -t 24:00:00 launch bash -x $LVSR/bin/run_cluster.sh train $@
}

# Copying task

[ -a actor_critic1 ] || run actor_critic1 $LVSR/exp/billion_words/configs/actor_critic1.yaml
[ -a actor_critic3 ] || run actor_critic3 $LVSR/exp/billion_words/configs/actor_critic3.yaml

# Denoising task

[ -a autoencoder2 ] || run autoencoder2 $LVSR/exp/billion_words/configs/autoencoder2.yaml

# Without DP
[ -a actor_critic4 ] || run actor_critic4 $LVSR/exp/billion_words/configs/actor_critic4.yaml
[ -a actor_critic4a ] || run actor_critic4a $LVSR/exp/billion_words/configs/actor_critic4.yaml\
                         training.catching_up_coof 0.001
[ -a actor_critic4b ] || run actor_critic4b $LVSR/exp/billion_words/configs/actor_critic4.yaml\
                         training.catching_up_coof 0.0001
[ -a actor_critic4c ] || (mkdir actor_critic4c;\
                          run actor_critic4c --params actor_critic4/critic_pretraining.tar --start-stage main $LVSR/exp/billion_words/configs/actor_critic4.yaml\
                          net.criterion.entropy_reward_coof 0.0)
[ -a actor_critic4d ] || (mkdir actor_critic4d;\
                          run actor_critic4d --params actor_critic4/critic_pretraining.tar --start-stage main $LVSR/exp/billion_words/configs/actor_critic4.yaml\
                          net.criterion.epsilon 0.0)
[ -a actor_critic4e ] || run actor_critic4e $LVSR/exp/billion_words/configs/actor_critic4.yaml net.criterion.same_value_for_wrong False


# With DP and frozen targets
[ -a actor_critic5 ] || run actor_critic5 $LVSR/exp/billion_words/configs/actor_critic5.yaml
[ -a actor_critic5a ] || run actor_critic5a $LVSR/exp/billion_words/configs/actor_critic5.yaml\
                         net.criterion.use_value_biases True
[ -a actor_critic5b ] || (mkdir actor_critic5b;\
                          run actor_critic5b --params actor_critic5/critic_pretraining.tar --start-stage main $LVSR/exp/billion_words/configs/actor_critic5.yaml\
                          net.criterion.entropy_reward_coof 0.0)
[ -a actor_critic5c ] ||  run actor_critic5c $LVSR/exp/billion_words/configs/actor_critic5.yaml\
                          net.criterion.use_value_biases False net.criterion.entropy_reward_coof 0.0

[ -a actor_critic6 ] || run actor_critic6 $LVSR/exp/billion_words/configs/actor_critic6.yaml
