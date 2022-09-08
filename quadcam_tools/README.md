# Quant for int8

# Generate TensorRT Engine
With DLA and int8
```bash
# Superpoint
python eval_cnns.py --model ../models/superpoint_v1.onnx --data-bchw --engine-cache ../models --width 400 --height 200 --enable-dla --enable-int8 #We use 400x200 on the Jetson Nx
# MobileNetVlad
python eval_cnns.py --model ../models/mobilenetvlad_240x320.onnx  --engine-cache ../models  --width 640 --height 480  --enable-dla --enable-int8  #Yes is now 480x640, will fix later.
```

Without DLA and int8
```bash
# Superpoint
python eval_cnns.py --model ../models/superpoint_v1.onnx --data-bchw --engine-cache ../models --width 400 --height 200#We use 400x200 on the Jetson Nx
# MobileNetVlad
python eval_cnns.py --model ../models/mobilenetvlad_240x320.onnx  --engine-cache ../models  --width 640 --height 480 #Yes is now 480x640, will fix later.
```
