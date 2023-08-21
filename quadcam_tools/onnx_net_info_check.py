# this scipt is used to check the input and output of onnx model
import onnxruntime
import argparse
import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='get model info')
  parser.add_argument('--model', type=str, help='path to model')
  args = parser.parse_args()
  model_path = args.model
  if not os.path.exists(model_path):
    print("model not exist\n")
    exit(1)
  print("model path: ", model_path)
  provider = []
  provider.append('CUDAExecutionProvider')
  sess = onnxruntime.InferenceSession(model_path, providers=provider)
  print(f"model {model_path} info:")
  for input in sess.get_inputs():
    print(f'input name: {input.name}, shape: {input.shape}')
  for output in sess.get_outputs():
    print(f'output name: {output.name}, shape: {output.shape}')
