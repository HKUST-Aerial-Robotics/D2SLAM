#include "swarm_loop/tensorrt_generic.h"
#include "swarm_loop/loop_defines.h"
#include "swarm_loop/utils.h"
#include "swarm_msgs/swarm_types.hpp"
using namespace Swarm;
using namespace nvinfer1;
uint64_t get3DTensorVolume4(nvinfer1::Dims inputDims);

TensorRTInferenceGeneric::TensorRTInferenceGeneric(std::string input_blob_name, int _width, int _height):
    m_InputBlobName(input_blob_name), width(_width), height(_height){

}

void TensorRTInferenceGeneric::init(const std::string & engine_path) {

    m_Engine = loadTRTEngine(engine_path, nullptr, m_Logger);
    assert(m_Engine != nullptr);
    
    m_Context = m_Engine->createExecutionContext();
	assert(m_Context != nullptr);

    for (unsigned int i = 0; i < m_Engine->getNbBindings(); i ++ ) {
        std::string name(m_Engine->getBindingName(i));
        std::cout  << "TensorRT binding index " << i << " name " << name << std::endl;
    }

	m_InputBindingIndex = m_Engine->getBindingIndex(m_InputBlobName.c_str());
	assert(m_InputBindingIndex != -1);
    std::cout << "MaxBatchSize" << m_Engine->getMaxBatchSize() << std::endl;
	assert(m_BatchSize <= static_cast<uint32_t>(m_Engine->getMaxBatchSize()));
	allocateBuffers();
	NV_CUDA_CHECK(cudaStreamCreate(&m_CudaStream));
	assert(verifyEngine());

    std::cout << "TensorRT workspace " << m_Engine->getWorkspaceSize () /1024.0/1024.0 << "mb" << std::endl;
}

void TensorRTInferenceGeneric::doInference(const cv::Mat & input) {
    // assert(input.channels() == 1 && "Only support 1 channel now");
    TicToc inference;
    //This function is very slow event on i7, we need to optimize it
    //But not now.
    if (input.channels() == 1) {
        doInference(input.data, 1);
    } else {
        cv::Mat bgr[3];
        cv::split(input, bgr);
        static float * data_buf = new float[3*input.rows*input.cols];
        memcpy(data_buf, bgr[2].data, input.rows*input.cols*sizeof(float));
        memcpy(data_buf+input.rows*input.cols, bgr[1].data, input.rows*input.cols*sizeof(float));
        memcpy(data_buf+input.rows*input.cols*2, bgr[0].data, input.rows*input.cols*sizeof(float));
        doInference((unsigned char*)data_buf, 1);
    }
    //printf("doInference %fms\n", inference.toc());
}


void TensorRTInferenceGeneric::doInference(const unsigned char* input, const uint32_t batchSize)
{
	//Timer timer;
    assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");
    NV_CUDA_CHECK(cudaMemcpyAsync(m_DeviceBuffers.at(m_InputBindingIndex), input,
                                  batchSize * m_InputSize * sizeof(float), cudaMemcpyHostToDevice,
                                  m_CudaStream));
	
    m_Context->enqueue(batchSize, m_DeviceBuffers.data(), m_CudaStream, nullptr);
    for (auto& tensor : m_OutputTensors)
    {
        NV_CUDA_CHECK(cudaMemcpyAsync(tensor.hostBuffer, m_DeviceBuffers.at(tensor.bindingIndex),
                                      batchSize * tensor.volume * sizeof(float),
                                      cudaMemcpyDeviceToHost, m_CudaStream));
    }
    cudaStreamSynchronize(m_CudaStream);
//	timer.out("inference");
}

bool TensorRTInferenceGeneric::verifyEngine()
{
    assert((m_Engine->getNbBindings() == (1 + m_OutputTensors.size())
            && "Binding info doesn't match between cfg and engine file \n"));

    for (auto tensor : m_OutputTensors)
    {
        assert(!strcmp(m_Engine->getBindingName(tensor.bindingIndex), tensor.blobName.c_str())
               && "Blobs names dont match between cfg and engine file \n");
        // std::cout << get3DTensorVolume4(m_Engine->getBindingDimensions(tensor.bindingIndex)) <<":" << tensor.volume << std::endl;
        assert(get3DTensorVolume4(m_Engine->getBindingDimensions(tensor.bindingIndex))
                   == tensor.volume
               && "Tensor volumes dont match between cfg and engine file \n");
    }

    assert(m_Engine->bindingIsInput(m_InputBindingIndex) && "Incorrect input binding index \n");
    assert(m_Engine->getBindingName(m_InputBindingIndex) == m_InputBlobName
           && "Input blob name doesn't match between config and engine file");
    assert(get3DTensorVolume4(m_Engine->getBindingDimensions(m_InputBindingIndex)) == m_InputSize);
    return true;
}

void TensorRTInferenceGeneric::allocateBuffers()
{
    m_DeviceBuffers.resize(m_Engine->getNbBindings(), nullptr);
    assert(m_InputBindingIndex != -1 && "Invalid input binding index");
    NV_CUDA_CHECK(cudaMalloc(&m_DeviceBuffers.at(m_InputBindingIndex),
                             m_BatchSize * m_InputSize * sizeof(float)));

    for (auto& tensor : m_OutputTensors)
    {
        tensor.bindingIndex = m_Engine->getBindingIndex(tensor.blobName.c_str());
        std::cout << "Tensor" << tensor.blobName.c_str() << " bind to " << tensor.bindingIndex 
                << " dim " << m_Engine->getBindingDimensions(tensor.bindingIndex).d[0]
                << " " << m_Engine->getBindingDimensions(tensor.bindingIndex).d[1]
                << " " << m_Engine->getBindingDimensions(tensor.bindingIndex).d[2]
                << " " << m_Engine->getBindingDimensions(tensor.bindingIndex).d[3] << std::endl;
        assert((tensor.bindingIndex != -1) && "Invalid output binding index");
        NV_CUDA_CHECK(cudaMalloc(&m_DeviceBuffers.at(tensor.bindingIndex),
                                 m_BatchSize * tensor.volume * sizeof(float)));
        NV_CUDA_CHECK(
            cudaMallocHost(&tensor.hostBuffer, tensor.volume * m_BatchSize * sizeof(float)));
    }
}


uint64_t get3DTensorVolume4(nvinfer1::Dims inputDims)
{
    int ret = 1;
    for (int i = 0; i < inputDims.nbDims; i ++) {
        ret = ret * inputDims.d[i];
    }
    return ret;
}
