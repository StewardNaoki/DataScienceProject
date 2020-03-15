set -e
num_epoch=1
num_batch=1
num_thread=1

python3 main.py --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log