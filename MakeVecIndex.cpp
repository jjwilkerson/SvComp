/**
 * @file
 * @brief Creates an index file corresponding to a file containing vectors to be compressed.
 *
 * Used to build an executable to build a vector index file, using a local config.json for configuration and loading
 * state from a local state.json file, if present. Enabled by building with -DPROGRAM_VERSION=2
 */

#if PROGRAM_VERSION == 2

#include <util/FileUtil.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace netlib;

typedef unsigned long long int ix_type;

void writeInt64(ix_type val, ofstream& file);

int main(int argc, char* argv[]) {
	if (argc < 4) {
		cerr << "format: vec_size num_vec out_filename" << endl;
		exit(-1);
	}

	int vecSize = atoi(argv[1]);
	int numVec = atoi(argv[2]);
	const char* filename = argv[3];

	int elementSize = 2;

	ofstream file(filename, ios::binary);

	FileUtil::writeInt64(numVec, file);
	for (int i = 0; i < numVec; i++) {
		uint64 ix = i * elementSize * vecSize;
		FileUtil::writeInt64(ix, file);
	}

	file.close();
}

void writeInt64(ix_type val, ofstream& file) {
	unsigned char buf[8];
	for (int i = 7; i >= 0; i--) {
		buf[i] = val & 0xff;
		val = val >> 8;
	}
	file.write(reinterpret_cast<char*>(buf), 8);
}

#endif
