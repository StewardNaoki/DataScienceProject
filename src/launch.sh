set -e
num_epoch=350
num_batch=32
num_thread=8
image_size=512

num_depth=6
num_block=3
python3 main.py --image_size $image_size --depth $num_depth --num_block $num_block --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log

num_depth=5
num_block=3
python3 main.py --image_size $image_size --depth $num_depth --num_block $num_block --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log

num_depth=6
num_block=2
python3 main.py --image_size $image_size --depth $num_depth --num_block $num_block --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log