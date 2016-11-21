#!/usr/bin/env bash

FROM=helios.calculquebec.ca:dist/fully-neural-lvsr/autoencoder4
COPY="rsync -rvzu --progress"

$COPY $FROM/\{actor_critic7i,actor_critic8i,actor_critic9i,actor_critic10i\} .
$COPY $FROM/\{actor_critic7j,actor_critic8j,actor_critic9j,actor_critic10j\} .
$COPY $FROM/\{actor_critic12a,actor_critic13a,actor_critic14a,actor_critic15a\} .
