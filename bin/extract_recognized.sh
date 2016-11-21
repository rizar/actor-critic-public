#!/usr/bin/env bash

 cat $1 | grep '^Recognized:' | sed 's/Recognized: \(.*\)/\1/'

