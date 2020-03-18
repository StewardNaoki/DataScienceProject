set -e
num_epoch=350
num_batch=32
num_thread=8
image_size=$1

depth=6
num_block=3
python3 main.py --image_size $image_size --depth $depth --num_block $num_block --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log

depth=5
num_block=3
python3 main.py --image_size $image_size --depth $depth --num_block $num_block --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log

depth=6
num_block=2
python3 main.py --image_size $image_size --depth $depth --num_block $num_block --epoch $num_epoch --batch $num_batch --num_thread $num_thread --log
