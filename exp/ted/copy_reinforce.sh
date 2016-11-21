set -x

for RUN in\
    reinforced2
do
    mkdir $RUN
    rsync -rvz helios.calculquebec.ca:dist/fully-neural-lvsr/mt3/$RUN/main.tar $RUN
done 


