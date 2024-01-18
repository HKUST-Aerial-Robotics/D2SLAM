SWARM_WS=/root/swarm_ws
DOCKERIMAGE="d2slam:pc"
LOCAL_WS=/path/to/your/local_dir
DATA_SET=/path/to/your/dataset

CONFIGS=${LOCAL_WS}/configs-drone-swarm

#deprecatred in realtime drone
HITNET=${LOCAL_WS}/perception-ONNX-HITNET-Stereo-Depth-estimation
CRESTEREO=${LOCAL_WS}/perception-ONNX-CREStereo-Depth-Estimation

#NN models
NN_MODELS=${LOCAL_WS}/NNmodels_generator

