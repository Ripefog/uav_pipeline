# Jetson — tạo venv + cài deps (1 lần).
# --system-site-packages để kế thừa tensorrt/pycuda cài sẵn qua JetPack apt
# (python3-libnvinfer...) — bản pip tensorrt/pycuda generic không chắc có
# wheel cho aarch64/Jetson, nên KHÔNG tự pip install 2 gói này trong venv.
uv venv .venv --system-site-packages
source .venv/bin/activate && uv pip install -r requirements.txt

# Jetson — export TensorRT engine (best_yoloxx, full precision, batch=4)
# Phải chạy TRÊN Jetson: engine build ở x86_64/RTX 5080 KHÔNG dùng được trên
# Jetson (khác kiến trúc CPU aarch64 vs x86_64, khác GPU arch sm_87 vs sm_120).
# batch=16 OOM trên Orin NX 8GB (full precision + 736x1280 quá nặng) -> batch=4.
source .venv/bin/activate
PYTHONPATH=/home/ftel-uav python -m uav_pipeline.scripts.export_tensorrt --onnx weights/best_yoloxx.onnx --engine weights/best_yoloxx.engine --imgsz 736 1280 --batch 4 --no-fp16

# Jetson — chạy đúng sequence uav0000339_00001_v (input/VisDrone2019-MOT-val),
# dùng engine vừa build (configs/jetson_trt.yaml, source override sang image_dir):
source .venv/bin/activate && PYTHONPATH=/home/ftel-uav python -m uav_pipeline.scripts.run_pipeline \
  -c configs/jetson_trt.yaml \
  --source-type image_dir \
  --source input/VisDrone2019-MOT-val/sequences/uav0000339_00001_v

# Hoặc dùng đúng gstreamer source đã khai trong config (RTSP/CSI gimbal cam),
# không cần override --source/--source-type:
source .venv/bin/activate && PYTHONPATH=/home/ftel-uav python -m uav_pipeline.scripts.run_pipeline \
  -c configs/jetson_trt.yaml
