/**
 * @file
 * @brief Entry point for single training run.
 *
 * Used to build an executable for a single training run, using a local config.json for configuration and loading
 * state from a local state.json file, if present. Enabled by building with -DPROGRAM_VERSION=1
 */

#if PROGRAM_VERSION == 1

#include <NetLib.h>
#include <input/VecIterator.h>
#include <config/Config.h>
#include <state/State.h>
#include "cublas_v2.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace netlib;

Config config;
const char* configFilename = "config.json";
const char* stateFilename = "state.json";

extern dtype2 run(string dirName, Config& config, State* state, VecIterator& trainCorpus, VecIterator& testCorpus,
		cublasHandle_t& handle, int maxIter = -1, int maxTimeS = 0, dtype2* scoreFinal = NULL);

int main(int argc, char* argv[]) {
	path curr(argc > 1 ? argv[1] : ".");
	current_path(curr);
	string sDir = current_path().string();
	const char* dataDirName = sDir.c_str();

	cout << "data dir: " << dataDirName << endl;

	path configFile(configFilename);

	if (exists(configFile)) {
		config.load(configFilename);
	} else {
		config.batchSize = 32;
		config.encoder_rnn_width = 2;
		config.decoder_rnn_width = 2;
		config.sentence_vector_width = 2;
		config.numBatchError = 1;
		config.numBatchGrad = 1; //70;
		config.numBatchG = 1; //4;
		config.pDropMatch = 0.0;
		config.seed = 1234;
		config.ffWeightCoef = 0.14150421321392059;
		config.recWeightCoef = 0.030431009829044342;
		config.l2 = 7e-04;
		config.initDamp = 11.449784278869629;
		config.structDampCoef = 1.8929900988950976e-06;
		config.optimizer = "adam";
	}

	State* state = NULL;

	path stateFile(stateFilename);
	if (exists(stateFile)) {
		state = new State();
		state->load(stateFilename);
	}

	if (state == NULL) {
		state = new State();
		state->ixFilename = datasetsDir + "sentvec.out.ix";
		state->curandStatesFilename = "";
		state->initDeltaFilename = "";
		state->epoch = 0;
		state->iter = -1;
		state->sentenceIx = 0;
		state->clock = 0;
	} else {
		config.weightsFilename = state->weightsFilename;
		config.curandStatesFilename = state->curandStatesFilename;

		config.initDamp = state->damping;
		config.deltaDecayInitial = state->deltaDecay;

		if (state->l2 >= 0) {
			config.l2 = state->l2;
		}

		config.numBatchGrad = state->numBatchGrad;
		config.numBatchG = state->numBatchG;
		config.numBatchError = state->numBatchError;

		if (state->maxIterCG > 0) {
			config.maxIterCG = state->maxIterCG;
		}

		if (state->learningRate > 0) {
			config.learningRate = state->learningRate;
		}

		config.lossScaleFac = state->lossScaleFac;
		config.iterNoOverflow = state->iterNoOverflow;
	}

	string ixFilename = state->ixFilename;
	ix_type startIx = state->sentenceIx;

	int batchSize = config.batchSize;
	int svSize = config.sentence_vector_width;
	int maxSeqLength = 1;
	config.maxSeqLength = maxSeqLength;

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	cout << endl << "Loading training corpus" << endl;
	string trainFilename = datasetsDir + "sentvec.out";
	VecIterator trainCorpus(trainFilename, svSize, batchSize, true, ixFilename, startIx);

	cout << endl << "Loading test corpus" << endl;
	string testFilename = datasetsDir + "sentvec_tune.out";
	VecIterator testCorpus(testFilename, svSize, batchSize, false);
	int maxIter = -1;
	int maxTimeS = 0;

	run(sDir, config, state, trainCorpus, testCorpus, handle, maxIter, maxTimeS);

    if (state != NULL) {
    	delete state;
    }

	cublasDestroy(handle);
}

#endif
