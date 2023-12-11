/**
 * @file
 * @brief Defines Trainer class.
 */

#include "Trainer.h"
#include <Network.h>
#include <config/Config.h>
#include <layers/Layer.h>
#include <optimizers/Optimizer.h>
#include <gpu/CublasFunc.h>
#include <state/State.h>
#include <state/IterInfo.h>
#include <loss/LossFunction.h>
#include <input/VecInputSource.h>
#include <input/VecIterator.h>
#include <util/FileUtil.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <ctime>
#include <boost/filesystem.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <utf8.h>

typedef boost::mt19937 base_generator_type;

using namespace std;
using namespace boost::filesystem;

const char* scoresFilename = "iter_scores";
const char* weightsTemp1Filename = "svcomp_weights_save.bin";
const char* weightsTemp2Filename = "svcomp_weights_save.bak.bin";
const char* initDeltaTemp1Filename = "svcomp_initDelta_save.bin";
const char* initDeltaTemp2Filename = "svcomp_initDelta_save.bak.bin";
const char* curandStatesTemp1Filename = "svcomp_curandStates_save.bin";
const char* curandStatesTemp2Filename = "svcomp_curandStates_save.bak.bin";
const char* stateTemp1Filename = "svcomp_state_save.json";
const char* stateTemp2Filename = "svcomp_state_save.bak.json";
const char* latestStateFilename = "state.json";
const char* iterInfoFilename = "iter_info";
const char* matchingFilename = "matching_counts";

const char* weightsEpochTemp1Filename = "svcomp_weights_epoch_save.bin";
const char* weightsEpochTemp2Filename = "svcomp_weights_epoch_save.bak.bin";
const char* initDeltaEpochTemp1Filename = "svcomp_initDelta_epoch_save.bin";
const char* initDeltaEpochTemp2Filename = "svcomp_initDelta_epoch_save.bak.bin";
const char* curandStatesEpochTemp1Filename = "svcomp_curandStates_epoch_save.bin";
const char* curandStatesEpochTemp2Filename = "svcomp_curandStates_epoch_save.bak.bin";
const char* stateEpochTemp1Filename = "svcomp_state_epoch_save.json";
const char* stateEpochTemp2Filename = "svcomp_state_epoch_save.bak.json";


namespace netlib {

Trainer::Trainer(cublasHandle_t& handle, Config& config, Network& net, Optimizer& optimizer, VecInputSource& trainSource,
		VecInputSource& testSource)
	: handle(handle), config(config), net(net), optimizer(optimizer), trainSource(trainSource), testSource(testSource) {

	ixFilenameBase = datasetsDir + "sv_ixs_epoch";
	int vecLength = trainSource.getVecLength();
	int length = config.batchSize * vecLength;
	int arraySize = length * sizeof(dtype2);
    checkCudaErrors(cudaMalloc((void **)&intermed, arraySize));
    checkCudaErrors(cudaMallocHost((void **)&h_output, arraySize));
}

Trainer::~Trainer() {
	checkCudaErrors(cudaFree(intermed));
	checkCudaErrors(cudaFreeHost(h_output));
}

dtype2 Trainer::train(string outputDirName, int startEpoch, int startIter, int maxIter, int maxTimeS, unsigned int clockOffset,
		int testPeriod, int savePeriod, int tempSavePeriod, int printPeriod, int maxEpochs, bool debug, dtype2* scoreFinal) {

	path outputDir(outputDirName);
	if (!exists(outputDir)) {
		create_directories(outputDir);
	}
	current_path(outputDir);

	long startTime = clock();

	cout << "maxIter: " << maxIter << endl;
	cout << "maxTimeS: " << maxTimeS << endl;

	dtype2 score = 0.0;
	int iterNum = startIter;
	bool lastIter = false;
	bool firstEpoch = true;
	long offTime = 0;

	bool testFirst = true;

	for (int ep = startEpoch; ep < maxEpochs && !lastIter; ep++) {
		cout << "========================================" << endl;
		cout << "Epoch " << ep << endl;

		time_t curtime;

//		cudaProfilerStart();

		bool epochEnd = false;

		while (!lastIter && !epochEnd) {
			IterInfo iterInfo(iterNum);

			bool printing = ((printPeriod > 0 && (iterNum % printPeriod == 0)) || debug);

#ifndef DEBUG
			if (testFirst && iterNum == 0) {
				long before = clock();

				dtypeh loss;
				score = test(testSource, iterNum, &loss);
				saveScore(iterNum, score, loss, 0);

				testFirst = false;
				long after = clock();
				offTime += (after - before) / CLOCKS_PER_SEC;
			}

#endif

			long now = clock();
			long elapsed = (now - startTime) / CLOCKS_PER_SEC;
			elapsed -= offTime;

			iterInfo.clock = elapsed + clockOffset;

			if ((maxIter >= 0 && iterNum >= maxIter) ||
				(maxTimeS > 0 && elapsed >= maxTimeS)) {
				lastIter = true;
			}

			if (printing) {
				time(&curtime);
				cout << endl << ctime(&curtime) << "Iteration " << iterNum << endl;

				cout << "sentence ix " << trainSource.firstBatchIx << endl;
			}
			iterInfo.sentenceIx = trainSource.firstBatchIx;

			trainSource.toFirstBatch();

//			net.resetSavedMasks();

			optimizer.computeUpdate(iterInfo, printing);

			long clockNow = ((clock() - startTime) / CLOCKS_PER_SEC) + clockOffset;
			clockNow -= offTime;

			int saveEpoch = ep;

			trainSource.toNextBatchSet();
			if (!trainSource.hasNextBatchSet()) {
				epochEnd = true;

				long before = clock();
				string ixFilename = buildIxFilename(ep + 1);
				path ixPath(ixFilename);
				if (exists(ixPath)) {
					cout << endl << "Loading corpus indices" << endl;
					trainSource.loadIxs(ixFilename);
				} else {
					cout << endl << "Shuffling training corpus" << endl;
					trainSource.shuffle();
					trainSource.saveIxs(ixFilename);
				}
				trainSource.reset();
				long after = clock();
				offTime += (after - before) / CLOCKS_PER_SEC;

				saveEpoch = ep + 1;
				save(saveEpoch, iterNum, trainSource.firstBatchIx, clockNow);
			}

			if (((iterNum % savePeriod == 0) && (iterNum != 0)) || lastIter) {
				save(saveEpoch, iterNum, trainSource.firstBatchIx, clockNow);
			}

			if ((iterNum % tempSavePeriod == 0) && (iterNum != 0)) {
				cout << endl << "offTime: " << offTime << endl;
				tempSave(saveEpoch, iterNum, trainSource.firstBatchIx, clockNow);
			}

			if (((iterNum % testPeriod == 0) && (iterNum != 0)) || lastIter || iterNum == 1000) {
				dtypeh loss;
				long before = clock();
				score = test(testSource, iterNum, &loss);
				saveScore(iterNum, score, loss, clockNow);
				long after = clock();
				offTime += (after - before) / CLOCKS_PER_SEC;
				//if best score, save score and W
				if (scoreFinal != NULL) {
					*scoreFinal = score;
				}
			}

			if (iterNum == 0) {
				IterInfo::saveHeader(iterInfoFilename);
			}
			if (printing) {
				iterInfo.save(iterInfoFilename);
			}

			iterNum++;
		}

		firstEpoch = false;
	}

	return score;
}

void Trainer::saveScore(int iterNum, dtype2 score, dtype2 loss, long clock) {
	std::ofstream file(scoresFilename, ios::app);
	file << fixed << setprecision(6);
	file << iterNum << '\t' << score << '\t' << loss << '\t' << clock << endl;
	file.close();
}

void Trainer::save(int epochNum, int iterNum, int sentenceIx, long clock) {
	static char buf[100];
	sprintf(buf, "svcomp_weights_iter%d.bin", iterNum);
	net.saveWeights(buf);
	string weightsFilename(buf);

	sprintf(buf, "svcomp_initDelta_iter%d.bin", iterNum);
	optimizer.saveInitDelta(buf);
	string initDeltaFilename(buf);

	sprintf(buf, "svcomp_curandStates_iter%d.bin", iterNum);
	net.saveCurandStates(buf);
	string curandStatesFilename(buf);

	sprintf(buf, "svcomp_state_iter%d.json", iterNum);
	saveState(buf, weightsFilename, initDeltaFilename, curandStatesFilename,
			epochNum, iterNum, sentenceIx, clock);

	path toPath(buf);
	path fromPath(latestStateFilename);
	if (exists(fromPath)) {
		remove(fromPath);
	}
	create_symlink(toPath, fromPath);
}

void Trainer::tempSave(int epochNum, int iterNum, int sentenceIx, long clock) {
	path weightsPath1(weightsTemp1Filename);
	path weightsPath2(weightsTemp2Filename);
	path idPath1(initDeltaTemp1Filename);
	path idPath2(initDeltaTemp2Filename);
	path csPath1(curandStatesTemp1Filename);
	path csPath2(curandStatesTemp2Filename);
	path sPath1(stateTemp1Filename);
	path sPath2(stateTemp2Filename);

	if (exists(weightsPath2)) {
		remove(weightsPath2);
	}
	if (exists(idPath2)) {
		remove(idPath2);
	}
	if (exists(csPath2)) {
		remove(csPath2);
	}
	if (exists(sPath2)) {
		remove(sPath2);
	}

	if (exists(weightsPath1)) {
		rename(weightsPath1, weightsPath2);
	}
	net.saveWeights(weightsTemp1Filename);

	if (exists(idPath1)) {
		rename(idPath1, idPath2);
	}
	optimizer.saveInitDelta(initDeltaTemp1Filename);

	if (exists(csPath1)) {
		rename(csPath1, csPath2);
	}
	net.saveCurandStates(curandStatesTemp1Filename);

	if (exists(sPath1)) {
		rename(sPath1, sPath2);
	}

	saveState(stateTemp1Filename, weightsTemp1Filename, initDeltaTemp1Filename, curandStatesTemp1Filename,
			epochNum, iterNum, sentenceIx, clock);

	path toPath(stateTemp1Filename);
	path fromPath(latestStateFilename);
	if (exists(fromPath)) {
		remove(fromPath);
	}
	create_symlink(toPath, fromPath);
}

void Trainer::saveState(const char* stateFilename, string weightsFilename, string initDeltaFilename,
		string curandStatesFilename, int epochNum, int iterNum, int sentenceIx, long clock) {
	State state;
	state.weightsFilename = weightsFilename;
	state.initDeltaFilename = initDeltaFilename;
	state.curandStatesFilename = curandStatesFilename;
	state.ixFilename = trainSource.getIxFilename();

	state.epoch = epochNum;
	state.iter = iterNum;
	state.sentenceIx = sentenceIx;
	state.damping = optimizer.getDamping();
	state.deltaDecay = optimizer.getDeltaDecay();
	state.l2 = net.l2;
	state.numBatchGrad = net.numBatchGrad;
	state.numBatchG = net.numBatchG;
	state.numBatchError = net.numBatchError;
	state.maxIterCG = optimizer.getMaxIterCG();
	state.clock = clock;
	state.learningRate = optimizer.getLearningRate();
	state.lossScaleFac = net.getLossScaleFac();
	state.iterNoOverflow = net.getIterNoOverflow();

	state.save(stateFilename);
}

string Trainer::buildIxFilename(int epochNum) {
	static char buf[100];
	sprintf(buf, "%s%d", ixFilenameBase.c_str(), epochNum);
	return string(buf);
}

void Trainer::epochCopyTempSaves() {
	path weightsPath1(weightsTemp1Filename);
	path weightsPath2(weightsTemp2Filename);
	path weightsEpochPath1(weightsEpochTemp1Filename);
	path weightsEpochPath2(weightsEpochTemp2Filename);
	if (exists(weightsPath1)) {
		copy(weightsPath1, weightsEpochPath1);
	}
	if (exists(weightsPath2)) {
		copy(weightsPath2, weightsEpochPath2);
	}

	path idPath1(initDeltaTemp1Filename);
	path idPath2(initDeltaTemp2Filename);
	path idEpochPath1(initDeltaEpochTemp1Filename);
	path idEpochPath2(initDeltaEpochTemp2Filename);
	if (exists(idPath1)) {
		copy(idPath1, idEpochPath1);
	}
	if (exists(idPath2)) {
		copy(idPath2, idEpochPath2);
	}

	path csPath1(curandStatesTemp1Filename);
	path csPath2(curandStatesTemp2Filename);
	path csEpochPath1(curandStatesEpochTemp1Filename);
	path csEpochPath2(curandStatesEpochTemp2Filename);
	if (exists(csPath1)) {
		copy(csPath1, csEpochPath1);
	}
	if (exists(csPath2)) {
		copy(csPath2, csEpochPath2);
	}

	path sPath1(stateTemp1Filename);
	path sPath2(stateTemp2Filename);
	path sEpochPath1(stateEpochTemp1Filename);
	path sEpochPath2(stateEpochTemp2Filename);
	if (exists(sPath1)) {
		copy(sPath1, sEpochPath1);
	}
	if (exists(sPath2)) {
		copy(sPath2, sEpochPath2);
	}
}

dtypeh Trainer::test(VecInputSource& source, int iterNum, dtypeh* loss) {
	cout << endl << "====================   TEST   ====================" << endl;

	static char buf[100];
	sprintf(buf, "svcomp_test_sv_iter%d.out", iterNum);
	std::ofstream svFile(buf, ios::binary | ios::trunc);

	int numBatches = 0;
	dtypeh totalError = 0.0;
	dtypeh totalLoss = 0.0;
	source.reset();
	int batchSize = source.getBatchSize();

	net.copyParams();

	source.reset();
	while (source.hasNext() && numBatches < config.testMaxBatches) {
		source.next();
		net.forward(0, NULL, false, false, &source, false);

		dtypeh err = squaredErrorBatch(batchSize);
		totalError += err;

		totalLoss += net.lossFunction.batchLoss(net.getOutputLayer()->activation, net.getTargets(), NULL,
				false, NULL, net.getDInputLengths()[0]);

		if (iterNum % 20000 == 0) {
			saveOutput(svFile);
		}

		numBatches++;
	}

	svFile.close();

	dtypeh avgErr = totalError / numBatches;
	dtypeh avgLoss = totalLoss / (batchSize * numBatches);
	cout << endl << "test error: " << avgErr << endl;
	cout << "test loss: " << avgLoss << endl;

	cout << endl << "==================== TEST END ====================" << endl;

	if (loss != NULL) {
		*loss = avgLoss;
	}
	return avgErr;
}

dtypeh Trainer::squaredErrorBatch(int batchSize) {
	dtype2* output = net.getOutputLayer()->activation[0];
	dtype2* target = net.getTargets()[0];

	int vecLength = trainSource.getVecLength();
	int length = batchSize * vecLength;
	int arraySize = length * sizeof(dtype2);

	checkCudaErrors(cudaMemcpy(intermed, output, arraySize, cudaMemcpyDeviceToDevice));
	CublasFunc::axpy(handle, length, &minus_one, target, 1, intermed, 1);

	dtypeh error;
	CublasFunc::dot(handle, length, intermed, 1, intermed, 1, &error);

	return error / (batchSize * 2);
}

void Trainer::saveOutput(std::ofstream& file) {
	dtype2* output = net.getOutputLayer()->activation[0];
	int batchSize = net.batchSize;
	int vecLength = trainSource.getVecLength();
	int length = batchSize * vecLength;
	int arraySize = length * sizeof(dtype2);

	checkCudaErrors(cudaMemcpy(h_output, output, arraySize, cudaMemcpyDeviceToHost));

#ifdef DEVICE_HALF
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < vecLength-1; i+=2) {
			int val;
			dtype2* d2p = (dtype2*) &val;
			d2p[0] = h_output[IDX2(b, i, batchSize)];
			d2p[1] = h_output[IDX2(b, i+1, batchSize)];
			FileUtil::writeInt(val, file);
		}
	}
#else
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < vecLength; i++) {
			dtype2 val = h_output[IDX2(b, i, batchSize)];
			dtypeh singleVal = d2float(val);
			FileUtil::writeFloat(singleVal, file);
		}
	}
#endif
}

} /* namespace netlib */
