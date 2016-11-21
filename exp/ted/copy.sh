#!/usr/bin/env bash

FROM=helios.calculquebec.ca:dist/fully-neural-lvsr/mt3
COPY="rsync -rvzu --progress"

#$COPY $FROM/\{ted16a,ted16e,ted16f,ted16g,ted16h,ted16i,ted16l,ted16k,ted16n,ted16m,ted16o\} .
#$COPY $FROM/\{ted16m,ted16m2,ted16m3,ted16m4,ted16m5\} .
$COPY $FROM/\{ted16a4,ted16m8,ted16m2,ted16o2\} .
$COPY $FROM/ted17_2 .
#$COPY $FROM/reinforced2 .
