
conf=$1
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sent_conv.py $conf