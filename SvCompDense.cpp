/**
 * @file
 * @brief Executes a single training run with a simple autoencoder to compress and decompress vectors.
 *
 * Builds a 2-layer densely-connected autoencoder network, trainer, and optimizer and executes a single training run. Takes
 * configuration, state, and other objects as input. Called from RunTrainSingle.cpp.
 */

#include <layers/VecInputLayer.h>
#include <input/VecInputSource.h>
#include <input/VecIterator.h>
#include <nonlinearity/Nonlinearity.h>
#include <layers/ActivationLayer2.h>
#include <layers/DenseLayer2.h>
#include <layers/ActivationLayer.h>
#include <layers/DenseLayer.h>
#include <layers/SplitLayer.h>
#include <layers/AddLayer.h>
#include <layers/TruncateLayer.h>
#include <layers/PadLayer.h>
#include <loss/LossFunction.h>
#include <config/WeightInit.h>
#include <config/Config.h>
#include <state/State.h>
#include <Network.h>
#include <optimizers/Adam.h>
#include <gpu/CublasFunc.h>
#include "Trainer.h"
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <boost/filesystem.hpp>

using namespace netlib;

string name(string base, int num);

dtype2 run(string dirName, Config& config, State* state, VecIterator& trainCorpus, VecIterator& testCorpus,
		cublasHandle_t& handle, int maxIter = -1, int maxTimeS = 0, dtype2* scoreFinal = NULL) {
	int batchSize = config.batchSize;
	int maxSeqLength = 1;
	int svSize = config.sentence_vector_width;

	int startEpoch = state->epoch;
	int startIter = state->iter + 1;
	ix_type startIx = state->sentenceIx;
	unsigned int clockOffset = state->clock;
	string initDeltaFilename = state->initDeltaFilename;

	Linear linear(batchSize);
	Tanh tanh(batchSize);
	LeakyReLU leakyRelu(batchSize);

	VecInputSource trainSource(trainCorpus, batchSize, svSize, 1, 1, startIx);
	VecInputSource testSource(testCorpus, batchSize, svSize, 1, 1, startIx);

	VecInputLayer inputLayer("input", handle, batchSize, svSize, &linear, 0.0f);

	LossFunction* loss = new SquaredError(batchSize, 1, handle);

	SparseWeightInit ffWeightInit(config.seed, config.ffWeightCoef, 0);
	SparseWeightInit recWeightInit(config.seed, config.recWeightCoef, 0);

	Network net(handle, batchSize, maxSeqLength, config, *loss, ffWeightInit, recWeightInit, true);
	net.setTrainSource(&trainSource);

	net.addInput(&inputLayer);

	DenseLayer dense1("dense1", handle, batchSize, 3000, 1, &leakyRelu, 0.0f);
	dense1.setPrev(&inputLayer);
	net.addHidden(&dense1);

	DenseLayer dense2("dense2", handle, batchSize, svSize, 1, &linear, 0.0f);
	dense2.setPrev(&dense1);
	net.addHidden(&dense2);

	dense2.asOutputLayer();
	net.setOutput(&dense2);

	net.init();

//	cout << endl << "sentence_ix: " << startIx << endl;
//
//	net.iterInit();
//	dtype2 error = net.error();
//	cout << "error: " << error << endl;
//
//	dtype2 *grad = net.calcGrad();
//	dtype2 norm;
//	CublasFunc::nrm2(handle, net.nParams, grad, 1, &norm);
//	cout << "grad norm: " << norm << endl;
//
////	Network::printStatsGpu("grad", grad, net.nParams);
//
//#ifdef DEBUG
//	net.checkGrad(grad);
//#endif

	Optimizer* optimizer = new Adam(handle, net, config, startIter, false, initDeltaFilename);

	Trainer trainer(handle, config, net, *optimizer, trainSource, testSource);

	dtype2 score = trainer.train(dirName, startEpoch, startIter, maxIter, maxTimeS, clockOffset, config.testPeriod,
			config.savePeriod, config.tempSavePeriod, config.printPeriod, 100, false, scoreFinal);

	delete optimizer;
	delete loss;
	return score;
}

string name(string base, int num) {
//	return base;
//	stringstream ss;
//	ss << base << num;
//	return ss.str().c_str();

	return base + "_" + to_string(num);
}
