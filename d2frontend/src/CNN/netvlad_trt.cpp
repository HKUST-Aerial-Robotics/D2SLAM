#include "d2frontend/CNN/netvlad_trt.h"
#include "d2frontend/CNN/CNN_generic.h"
#include <stdint.h>
#include "d2frontend/utils.h"


namespace TensorRTNetVLAD{

NetVLADTrt::NetVLADTrt(int32_t stream_number, int32_t width, int32_t hight,
 std::string & engine_path, std::string& pca_table_path, bool show_info): stream_number_(stream_number),
  width_(width), height_(hight), engine_path_(engine_path), pca_table_path_(pca_table_path), show_info_(show_info){
    if (stream_number_ <= 0){
      stream_number = 1;
    }
    if (width_ <= 0){
      width_ = 300;
    }
    if (height_ <= 0){
      height_ = 150;
    }
    return;
};

int32_t NetVLADTrt::init(){
  if(this->pca_comp_path_.empty()){
    printf("[NetVLADTrt] pca_comp_path is empty; PCA disabled\n");
    pca_comp_T_.resize(1, 1);
    pca_mean_.resize(1, 1);
  } else {
    printf("Should not be here\n");
    auto pca = D2FrontEnd::load_csv_mat_eigen(this->pca_comp_path_);
    pca_mean_ = pca.row(0).transpose();
    pca_comp_T_ = pca.bottomRows(pca.rows() - 1).transpose();
    printf("PCA comp shape %d x %d mean %d\n", pca_comp_T_.rows(), pca_comp_T_.cols(), pca_mean_.rows());
  }
  nv_runtime_ptr_ = nvinfer1::createInferRuntime(this->logger_);
  if(nv_runtime_ptr_ == nullptr){
    std::cout << "createInferRuntime failed" << std::endl;
    return -1;
  }
  if(this->engine_path_.empty()){
    std::cout << "engine path is empty" << std::endl;
    return -2;
  }
  std::ifstream engine_file(this->engine_path_, std::ios::binary);
  if(!engine_file){
    std::cout << "open engine file failed" << std::endl;
    return -3;
  }
  std::stringstream engine_buffer;
  engine_buffer << engine_file.rdbuf();
  std::string plan = engine_buffer.str();
  printf("[NetVLADTRT] engine plan size %d\n", plan.size());
  auto engine = nv_runtime_ptr_->deserializeCudaEngine(plan.data(), plan.size(), nullptr);
  if (engine == nullptr) {
    std::cout << "deserializeCudaEngine failed" << std::endl;
    return -4;
  }
  //create engine
  std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr(engine, InferDeleter());
  nv_engine_ptr_ = engine_ptr;
  if(nv_engine_ptr_ == nullptr){
    std::cout << "create engine failed" << std::endl;
    return -5;
  }
  //create executors
  for(int32_t i = 0; i < stream_number_; i++){
    NetVLADExcutor executor;
    int32_t ret = executor.init(nv_engine_ptr_, input_tensor_names_, output_tensor_names_);
    if(ret != 0){
      std::cout << "init executor failed" << std::endl;
      return -6;
    }
    excutor_vec_.push_back(executor);
  }
  //init descriptors memory
  descriptors_.resize(stream_number_);
  printf("[NetVLADTrt] init done\n");
  return 0;
};

int32_t NetVLADTrt::doInference(std::vector<cv::Mat>& inputs){
  if(inputs.size() != stream_number_){
    std::cout << "input size is not equal to stream number" << std::endl;
    return -1;
  }
  for(int32_t i = 0; i < stream_number_; i++){
    int32_t ret = excutor_vec_[i].setInputImages(inputs[i]);
    if(ret != 0){
      std::cout << "set input images failed" << std::endl;
      return -2;
    }
  }
  for(int32_t i = 0; i < stream_number_; i++){
    int32_t ret = excutor_vec_[i].doInference();
    if(ret != 0){
      std::cout << "do inference failed" << std::endl;
      return -3;
    }
  }
  for(int32_t i = 0; i < stream_number_; i++){
    int32_t ret = excutor_vec_[i].copyBack();
    if(ret != 0){
      std::cout << "copy back failed" << std::endl;
      return -4;
    }
  }
  for(int32_t i = 0; i < stream_number_; i++){
    int32_t ret = excutor_vec_[i].synchronize();
    if(ret != 0){
      std::cout << "synchronize failed" << std::endl;
      return -5;
    }
  }
  return 0;
};

//TODO: 27 Nov 2023 compromise for API
int32_t NetVLADTrt::getOutput(std::vector<std::vector<float>> & descriptors){
  descriptors.clear();
  descriptors.resize(stream_number_);
  for(int32_t i = 0; i < stream_number_; i++){
    int32_t ret = excutor_vec_[i].getOutput(descriptors_[i]);
    if(ret != 0){
      std::cout << "get output failed" << std::endl;
      return -1;
    }
    if (pca_comp_T_.rows() > 1) {
      Eigen::Map<Eigen::VectorXf> desc(descriptors_[i].data(), descriptors_[i].size());
      Eigen::VectorXf desc_pca = pca_comp_T_ * (desc - pca_mean_);
      desc_pca /= desc_pca.norm();
      descriptors.push_back(std::vector<float>(desc_pca.data(), desc_pca.data() + desc_pca.size()));
    } else {
      descriptors.push_back(std::vector<float>(descriptors_[i].data(), descriptors_[i].data() + descriptors_[i].size()));
    }
  }
  return 0;
};

NetVLADTrt::~NetVLADTrt(){
  if(nv_runtime_ptr_ != nullptr){
    nv_runtime_ptr_ = nullptr;
  }
  if(nv_engine_ptr_ != nullptr){
    nv_engine_ptr_ = nullptr;
  }
};

int32_t NetVLADExcutor::init(std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr, 
    std::vector<std::string> input_tensor_names,
    std::vector<std::string> output_tensor_names){
  engine_ptr_ = engine_ptr;
  if(engine_ptr_ == nullptr){
    std::cout << "engine ptr is nullptr" << std::endl;
    return -1;
  }
  //create context
  nv_context_ptr_ = engine_ptr_->createExecutionContext();
  if(nv_context_ptr_ == nullptr){
    std::cout << "create context failed" << std::endl;
    return -2;
  }
  cudaStreamCreate(&stream_);
  //create buffer manager
  buffer_manager_ptr_ = std::make_shared<CudaMemoryManager::BufferManager>(engine_ptr_, 1, nv_context_ptr_);
  if(buffer_manager_ptr_ == nullptr){
    std::cout << "create buffer manager failed" << std::endl;
    return -2;
  }
  input_tensor_names_ = input_tensor_names;
  output_tensor_names_ = output_tensor_names;
  auto ret = initTensorMap(input_tensor_names_, input_tensor_map_);
  if(ret != 0){
    std::cout << "init input tensor map failed" << std::endl;
    return -2;
  }
  ret = initTensorMap(output_tensor_names_, output_tensor_map_);
  if(ret != 0){
    std::cout << "init output tensor map failed" << std::endl;
    return -3;
  }
  // printf("GDB debug here\n");
  return 0;
};

int32_t NetVLADExcutor::setInputImages(const cv::Mat& input){
  void* host_buffer = buffer_manager_ptr_->getHostBuffer(input_tensor_names_[0]);
  if (input.data == nullptr) {
    std::cout << "input data is nullptr" << std::endl;
    return -1;
  }
  if(host_buffer == nullptr){
    std::cout << "[NetVLADExcutor]get host buffer failed" << std::endl;
    return -2;
  }
  memcpy(buffer_manager_ptr_->getHostBuffer(input_tensor_names_[0]),
    input.data, input_tensor_map_["image:0"].binding_size);
  
  buffer_manager_ptr_->copyInputToDeviceAsync(stream_);
  return 0;
};

int32_t NetVLADExcutor::doInference(){
  //TODO:  dump here
  auto debug = buffer_manager_ptr_->getDeviceBindings().data();
  bool status = nv_context_ptr_->enqueueV2(debug ,stream_, nullptr);
  if (!status) {
    std::cout << "enqueueV2 failed" << std::endl;
    return -1;
  }
  return 0;
};

int32_t NetVLADExcutor::copyBack(){
  buffer_manager_ptr_->copyOutputToHostAsync(stream_);
  return 0;
};

int32_t NetVLADExcutor::synchronize(){
  cudaStreamSynchronize(stream_);
  return 0;
};

int32_t NetVLADExcutor::getOutput(std::array<float, kNetVLADDesRawLen>& descriptors){
  void* host_buffer = buffer_manager_ptr_->getHostBuffer(output_tensor_names_[0]);
  if(host_buffer == nullptr){
    std::cout << "[NetVLADExcutor]get host buffer failed" << std::endl;
    return -1;
  }
  memcpy(descriptors.data(), host_buffer, output_tensor_map_["descriptor:0"].binding_size);
  return 0;
};

int32_t NetVLADExcutor::getTensorSize(std::string & tensor_name){
  auto tensor = engine_ptr_->getBindingDimensions(output_tensor_map_[tensor_name].binding_index);
  if(tensor.nbDims == 0){
    std::cout << "get tensor size failed" << std::endl;
    return -1;
  }
  int32_t size = 1;
  for(int32_t i = 0; i < tensor.nbDims; i++){
    size *= tensor.d[i];
  }
  return size;
};

int32_t NetVLADExcutor::initTensorMap(std::vector<std::string> tensor_names,
    TensorMap & tensor_map){
  for(int32_t i = 0; i < tensor_names.size(); i++){
    auto index = engine_ptr_->getBindingIndex(tensor_names[i].c_str());
    if(index < 0){
      std::cout << "get input binding index failed" << std::endl;
      return -4;
    }
    auto input_dim = engine_ptr_->getBindingDimensions(index);
    int32_t size = input_dim.d[0] * input_dim.d[1] * input_dim.d[2] * input_dim.d[3] *sizeof(float);
    if(size < 0){
      std::cout << "get input binding size failed" << std::endl;
      return -5;
    }
    NodeInfo node_info;
    node_info.tensor_name = tensor_names[i];
    node_info.binding_index = index;
    node_info.binding_size = size;
    tensor_map[tensor_names[i]] = node_info;
  }
  return 0;
};
}