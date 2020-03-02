#!/bin/bash - 
#===============================================================================
#
#          FILE: test_multi.sh
# 
#         USAGE: ./test_multi.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 01/10/2020 09:36
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

ROOT="/media/feit/Work/Work/SpeakerID/Kaldi_Voxceleb/exp_clean/xvector_tf_but_test"
rm -r ${ROOT}/xvectors_train/
rm -r ${ROOT}/xvectors_voxceleb1_test/
rm ${ROOT}/model_final

for f in 140 280 420 560
do
	curr="model_${f}"
	ln -s ${ROOT}/${curr} ${ROOT}/model_final
	./run.sh 1>xvector_clean1_${f}.log 2>&1
	rm -r ${ROOT}/xvectors_train/
	rm -r ${ROOT}/xvectors_voxceleb1_test/
	rm ${ROOT}/model_final
done
