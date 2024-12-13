# Quant for int8

# Generate TensorRT Engine
With DLA and int8 (for accelerate on Xaiver NX)
```bash
# Superpoint
python3 eval_cnns.py --model ../models/superpoint_series/superpoint_v1_dyn_size.onnx --data-bchw --engine-cache ../models --width 400 --height 200 --enable-dla --enable-int8 --calib-table --enable-fp16 --enable-tensorrt ../models/superpoint_calibration.flatbuffers #We use 400x200 on the Jetson Nx
# MobileNetVlad
python eval_cnns.py --model ../models/mobilenetvlad_240x320.onnx  --engine-cache ../models  --width 400 --height 200  --enable-dla --enable-int8
```

Without DLA and int8
```bash
# Superpoint
python3 eval_cnns.py --model ../models/superpoint_series/superpoint_v1_dyn_size.onnx --data-bchw --engine-cache ../models --width 400 --height 200 #We use 400x200 on the Jetson Nx
# MobileNetVlad
python3 eval_cnns.py --model ../models/netvlad_series/mobilenetvlad_dyn_size.onnx  --engine-cache ../models  --width 400 --height 200
```
