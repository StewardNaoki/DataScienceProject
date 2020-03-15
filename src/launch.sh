set -e
num_epoch=350
num_batch=32
num_thread=8

python3 main.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log