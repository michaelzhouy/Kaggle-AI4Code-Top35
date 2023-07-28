cd train
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train.py >> train.txt

# cd ../valid
# export CUDA_VISIBLE_DEVICES="2,3"
# python3 valid.py > valid.txt

# cd ..
