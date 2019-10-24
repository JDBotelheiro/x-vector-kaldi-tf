#!/bin/bash

:<<EOF
for i in {2..40}
do
	perl local/contrib/change_dir_inscp.pl /home/feit/Tool/kaldi/egs/voxceleb/v2 /media/feit/Data/Archive/Home/Tool/kaldi/egs/voxceleb/v2 /media/feit/Data/Archive/Home/Tool/kaldi/egs/voxceleb/v2/data/voxceleb1_test/split40utt/$i/feats.scp /media/feit/Data/Archive/Home/Tool/kaldi/egs/voxceleb/v2/data/voxceleb1_test/split40utt/$i/vad.scp
done
EOF

perl local/contrib/change_dir_inscp.pl /home/feit/Tool/kaldi/egs/voxceleb/v2 /media/feit/Data/Archive/Home/Tool/kaldi/egs/voxceleb/v2 /media/feit/Data/Archive/Home/Tool/kaldi/egs/voxceleb/v2/data/voxceleb1_test/feats.scp /media/feit/Data/Archive/Home/Tool/kaldi/egs/voxceleb/v2/data/voxceleb1_test/vad.scp
