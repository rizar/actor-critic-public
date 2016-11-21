#!/usr/bin/env bash

grep Recognized: | sed 's/Recognized: \(.*\)/\1/'
