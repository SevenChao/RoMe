export WANDB_BASE_URL="http://localhost:8080"
export WANDB_API_KEY="local-key-123456789012345678901234567890"
export WANDB_MODE="offline"
export PYTHONPATH=${PWD}
export CUDA_VISIBLE_DEVICES=0

# python3 scripts/train.py --config configs/local_nusc_mini.yaml
python3 scripts/train.py --config configs/local_nusc.yaml
# python3 scripts/train.py --config configs/local_kitti.yaml