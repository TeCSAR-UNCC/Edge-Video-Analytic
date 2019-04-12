#ifndef __TRT_HELPER_HPP__
#define __TRT_HELPER_HPP__

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "/usr/src/tensorrt/samples/common/common.h"

using namespace nvinfer1;
IExecutionContext* context;
static Logger gLogger;
static int gUseDLACore{-1};

/********************************************************/
/********************************************************/
/***************TensorRT Inference Functions*************/
/********************************************************/
/********************************************************/
const std::vector<std::string> directories{"/usr/src/tensorrt/data/samples/mnist/", "/usr/src/tensorrt/data/mnist/models/", "/home/tecsar/Edge-Video-Analytic/models/reid/"};
std::string locateFile(const std::string& input)
{
	return locateFile(input, directories);
}

void onnxToTRTModel(const std::string& modelFile, // name of the onnx model
	                unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
	                IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
	int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	auto parser = nvonnxparser::createParser(*network, gLogger);

	//Optional - uncomment below lines to view network layer information
	//config->setPrintLayerInfo(true);
	//parser->reportParsingInfo();

	if (!parser->parseFromFile(locateFile(modelFile, directories).c_str(), verbosity))
	{
	    string msg("failed to parse onnx file");
	    gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
	    exit(EXIT_FAILURE);
	}

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	/*************************/
	builder->setFp16Mode(true);
	builder->allowGPUFallback(true);
	/**************************/

	samplesCommon::enableDLA(builder, gUseDLACore);
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we can destroy the parser
	parser->destroy();

	// serialize the engine, then close everything down
	trtModelStream = engine->serialize();
	engine->destroy();
	network->destroy();
	builder->destroy();
}

#endif //__TRT_HELPER_HPP__
