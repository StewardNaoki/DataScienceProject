set -e
num_epoch=1
num_batch=1
num_thread=1
image_size=64

num_depth=2
num_block=2
python3 main.py --image_size $image_size --depth $num_depth --num_block $num_block --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log