#!/bin/bash - 
#===============================================================================
#
#          FILE: debug_clean.sh
# 
#         USAGE: ./debug_clean.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 10/14/2019 11:14
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

expdir=/media/feit/Work/Work/SpeakerID/Kaldi_Voxceleb/exp_clean/xvector_tf_but_test
rm -r $expdir/log/ 1>/dev/null 2>&1
rm -r $expdir/model_*/ 1>/dev/null 2>&1
rm -r $expdir/xvectors_*/ 1>/dev/null 2>&1
rm $expdir/accuracy.report 1>/dev/null 2>&1
rm $expdir/model_final 1>/dev/null 2>&1
