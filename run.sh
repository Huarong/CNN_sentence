
conf=$1
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python sent_conv.py $conf
