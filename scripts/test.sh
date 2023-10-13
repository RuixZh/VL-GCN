CUDA_LAUNCH_BLOCKING=1
now=$(date +"%Y%m%d_%H%M%S")

# torchrun \
# 	--nnodes=1 \
# 	--nproc_per_node=1 \
python3	train.py --sim_header "Transf" --batch_size 64 --max_frames 8 --eval_freq 1 --log_time $now
