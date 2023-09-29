# reference:https://zhuanlan.zhihu.com/p/402074214

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import argparse
import numpy as np

class HostDeviceMem(object):
  def __init__(self, host_mem, device_mem):
      self.host = host_mem
      self.device = device_mem

  def __str__(self):
      return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

  def __repr__(self):
      return self.__str__()
  
class StreamContext(object):
  def __init__(self,inputs,outputs,bindings,stream,conext):
    self.inputs_ = inputs
    self.outputs_ = outputs
    self.bindings_ = bindings
    self.stream_ = stream
    self.context_ = conext

  def getInputs(self):
    return self.inputs_
  
  def setInputs(self,inputs):
    self.inputs_ = inputs
  
  def setInpustV3(self,inputs):
    pass


  def doAsyncInference(self):
    [cuda.memcpy_htod_async(inp.device, inp.host, self.stream_) for inp in self.inputs_]
    self.context_.execute_async_v2(bindings=self.bindings_, stream_handle=self.stream_.handle)
    # self.context_.execute_async_v3(stream_handle=self.stream_.handle)
    # self.context_.execute_async_v3(self.inputs_., self.stream_.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, self.stream_) for out in self.outputs_]

  def waitForSync(self):
    self.stream_.synchronize()
    # cuda.cudaStreamSynchronize(self.stream_)

  def getOutputs(self):
    return [out.host for out in self.outputs_]

class InfereceExcutor:
  def __init__(self, engine, streams_num):
    self.streams_num_ = streams_num
    self.stream_contexts_ = []
    self.engine_ = engine
    # print(f"${engine.get_binding_name()}")
    # new_context = engine.create_execution_context()
    for i in range(streams_num):
      # genereate context and streams
      new_context = engine.create_execution_context()
      new_stream = cuda.Stream()
      # allocate buffers
      inputs,outputs ,bindings = self.__allocateMemory(engine)
      self.stream_contexts_.append(StreamContext(inputs,outputs,bindings,new_stream,new_context))

  # input left_image and right_image
  def prepareData(self):
    for stream_context in self.stream_contexts_:
      inputs = stream_context.getInputs()
      for inp in inputs:
        inp.host = np.random.random(inp.host.shape).astype(inp.host.dtype)
      stream_context.setInputs(inputs)

  def startAsyncInfference(self):
    for stream_context in self.stream_contexts_:
      stream_context.doAsyncInference()
    for steam_context in self.stream_contexts_:
      steam_context.waitForSync()
    

  def getResults(self):
    out_puts = []
    for stream_context in self.stream_contexts_:
      out_puts.append(stream_context.getOutputs())
    return out_puts
    
  def __allocateMemory(self,engine):
    inputs = []
    outputs = []
    bindings = []
    for binding in engine:
      print(binding)
      size = trt.volume(engine.get_binding_shape(binding)) * 1
      dtype = trt.nptype(engine.get_binding_dtype(binding))
      #allocate host and device memory
      host_mem = cuda.pagelocked_empty(size, dtype)
      device_mem = cuda.mem_alloc(host_mem.nbytes)
      bindings.append(int(device_mem))
      if engine.binding_is_input(binding):
        inputs.append(HostDeviceMem(host_mem, device_mem))
      else:
        outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Create stereo Tensor engine test\n")
  parser.add_argument("--trt", type=str, default="", help="Path to engine file")
  parser.add_argument("--streams", type=int, default=1, help="Number of streams")
  parser.add_argument("--iterations", type=int, default=120, help="Number of iterations")
  args = parser.parse_args()
  streams_num = args.streams
  engine_path = args.trt
  engine = None
  cre_logger = trt.Logger(trt.Logger.INFO)
  trt.init_libnvinfer_plugins(None,'')
  profile = trt.Profiler()

  if (os.path.exists(engine_path)):
    with open(engine_path, 'rb') as f, trt.Runtime(cre_logger) as runtime:
      engine = runtime.deserialize_cuda_engine(f.read())
  else:
    print(f"Engine file ${engine_path} not found")
  excutor =  InfereceExcutor(engine,streams_num)
  excutor.prepareData()
  start_event = cuda.Event()
  end_event = cuda.Event()
  start_event.record()
  # warm up
  for i in range(100):
    excutor.startAsyncInfference()
    excutor.getResults()
  end_event.record()
  end_event.synchronize()
  elapsed_time_ms = start_event.time_till(end_event)/100
  print("Warm up time: {:.2f} ms".format(elapsed_time_ms))
  print("Warm up finished start to test")

  start_event = cuda.Event()
  end_event = cuda.Event()
  start_event.record()
  for i in range(args.iterations):
    excutor.startAsyncInfference()
  end_event.record()
  end_event.synchronize()
  runtime_per = start_event.time_till(end_event)/args.iterations
  print("Running time: {:.2f} ms".format(runtime_per))
  results = excutor.getResults()