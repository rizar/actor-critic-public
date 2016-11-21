#!/usr/bin/env bash
# Preprocess TED data
# Arguments
# 1 - the path on the raw data
# 2 - the destination to write to

# Step 1: get rid of tags
filetype=`file -b $1`
if [ "$filetype" == 'XML document text' ];
then
    # If it an XML file, take all "segments"
    grep '<seg id=.*>.*</seg>' $1  | sed 's/<seg.*>\(.*\)<\/seg>/\1/g' >step1.tmp
else
    # If not, take all lines, except for meta-info
    grep -v '<.*>' $1 >step1.tmp
fi
tr '[:upper:]' '[:lower:]' <step1.tmp >step2.tmp
$MOSES/scripts/tokenizer/tokenizer.perl <step2.tmp >$2

rm step1.tmp step2.tmp
