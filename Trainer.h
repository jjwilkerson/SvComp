/**
 * @file
 * @brief Declares Trainer class
 *
 */

#ifndef TRAINER_H_
#define TRAINER_H_

#include <NetLib.h>
#include "cublas_v2.h"
#include <string>
#include <map>
#include <vector>

using namespace std;

namespace netlib {

class Config;
class Network;
class Optimizer;
class VecInputSource;

/**
 * @brief Trainer class
 *
 * Trains the given network with the given optimizer and other objects.
 */
class Trainer {
public:
	Trainer(cublasHandle_t& handle, Config& config, Network& net, Optimizer& optimizer, VecInputSource& trainSource,
			VecInputSource& testSource);
	virtual ~Trainer();
	dtype2 train(string outputDirName, int startEpoch = 0, int startIter = 0, int maxIter = -1, int maxTimeS = 0,
			unsigned int clockOffset = 0, int testPeriod = 2000, int savePeriod = 10000, int tempSavePeriod = 200,
			int printPeriod = 20, int maxEpochs = 100, bool debug = false, dtype2* scoreFinal = NULL);
private:
	string ixFilenameBase;
	cublasHandle_t& handle;
	Config& config;
	Network& net;
	Optimizer& optimizer;
	VecInputSource& trainSource;
	VecInputSource& testSource;
	void saveScore(int iterNum, dtype2 score, dtype2 loss, long clock);
	void save(int epochNum, int iterNum, int sentenceIx, long clock);
	void tempSave(int epochNum, int iterNum, int sentenceIx, long clock);
	void saveState(const char* stateFilename, string weightsFilename, string initDeltaFilename,
			string curandStatesFilename, int epochNum, int iterNum, int sentenceIx, long clock);
	string buildIxFilename(int epochNum);
	void epochCopyTempSaves();
	dtypeh test(VecInputSource& source, int iterNum, dtypeh* loss = NULL);
	dtypeh squaredErrorBatch(int batchSize);
	void saveOutput(std::ofstream& file);

	dtype2* intermed;
	dtype2* h_output;
};

} /* namespace netlib */

#endif /* TRAINER_H_ */
