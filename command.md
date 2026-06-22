# OpenVINO batch-16 (~22.7 FPS):
source .venv/bin/activate && PYTHONPATH=/home/anlnm/UAV python -m uav_pipeline.scripts.run_pipeline -c configs/local_openvino_batch16.yaml --source "input/LG Revolution 720p HD Video Sample [B4qN3_KwpSw].mp4"

# ONNX batch-16 (~8.1 FPS):
source .venv/bin/activate && PYTHONPATH=/home/anlnm/UAV python -m uav_pipeline.scripts.run_pipeline -c configs/local_onnx_batch16.yaml --source "input/LG Revolution 720p HD Video Sample [B4qN3_KwpSw].mp4"