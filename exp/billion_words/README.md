The baseline experiments with log-likelihood can be reproduced by running ``autoencoder3-6.yaml`` configurations,
for example

``$LVSR/bin/run.py train autoencoder3 autoencoder3.yaml``

The actor-critic experiments used in the paper are ``actor_critic7-10.yaml`` configurations. For REINFORCE-critic, use ``actor_critic12-15.yaml``.

To reproduce the results without the critic having access to
the states of the actor, add ``net.criterion.critic_uses_actor_states False`` as two last command line arguments when you run ``$LVSR/bin/run.py train ...``.

