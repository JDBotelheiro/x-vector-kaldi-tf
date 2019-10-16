#!/bin/bash - 
#===============================================================================
#
#          FILE: read_log.sh
# 
#         USAGE: ./read_log.sh 
# 
#   DESCRIPTION: read the latest training log
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 10/15/2019 18:25
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

exp_dir=/media/feit/Work/Work/SpeakerID/Kaldi_Voxceleb/exp_ladder/xvector_tf_but_test/log
latest=0
for f in $exp_dir/train.*.log
do
	filename=$(basename -- "$f")
	IFS="."
	read -ra FILENAME <<< "$filename"
	curr=$((FILENAME[1]))
	if ((  curr >= latest )); then
		latest=$((curr))
	fi
done

latest_log="$exp_dir/train.$latest.1.log"

echo "$latest_log"
vim "$latest_log"
