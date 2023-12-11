# SvComp

## Overview

SvComp is a specialized software module designed to complement the SentEncDec project. Its primary function is to compress and decompress vectors, specifically tailored for handling the vector outputs produced by SentEncDec. This module plays a critical role in the workflow of encoding and decoding sentences, enabling efficient manipulation and storage of vector representations.

This project is associated with the paper titled "Return of the RNN: Residual Recurrent Networks for Invertible Sentence Embeddings" which provides in-depth explanations of the concepts, methodologies, and findings that underpin this software. The paper can be accessed [here](https://arxiv.org/abs/2303.13570v2).

For detailed documentation, see [SvComp Documentation](https://jjwilkerson.github.io/SvComp/).

## Compatibility

This software has been developed and tested on Linux. While it may work on other UNIX-like systems, its compatibility with non-Linux operating systems (like Windows or macOS) has not been verified. Users are welcome to try running it on other systems, but should be aware that they may encounter issues or unexpected behavior.

## Dependencies
To build and use SvComp, you'll need the following dependencies:

- **C++ Compiler:** GCC (versions 6.x - 12.2) or an equivalent compiler.
- **CUDA Toolkit:** Version 12.1 - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
- **Additional Libraries:**
    - NetLib (another module next to this one).
    - UTF8-CPP (version 2.3.4) [Download UTF8-CPP](https://github.com/nemtrif/utfcpp/tree/v2.3.4).
    - JsonCpp (version 1.8.4) [Download JsonCpp](https://github.com/open-source-parsers/jsoncpp/tree/1.8.4).
    - Boost (version 1.72.0) [Download Boost](https://www.boost.org/users/history/).
        - Build the regex, system, and filesystem modules.
    - CUDA Samples Common [Download CUDA Samples](https://github.com/NVIDIA/cuda-samples).

Ensure you have these libraries installed and accessible in your development environment.

## Configuring Dependencies in Eclipse
If building with Eclipse, it is necessary to configure the dependencies. This involves setting up include paths, library paths, and linking the libraries. Here's how to do it:

### Include Paths
- Add the include paths for NetLib, UTF8-CPP, and CUDA Samples Common
- Example paths:
    - ${workspace_loc:/NetLib}
    - /usr/local/include/utf8
    - /usr/local/cuda-samples/Common

### Library Paths
- Specify the library paths where Eclipse can find the compiled libraries.
- Example paths:
    - ${workspace_loc:/NetLib/Release}
    - /usr/local/lib (location of JsonCpp library)
    - /usr/lib/x86_64-linux-gnu (location of Boost libraries
    - /usr/local/cuda-12.1/lib64 (location of CUDA libraries

### Libraries
- Link the libraries in your Eclipse project settings.
- Example libraries to link:
    - NetLib
    - boost_regex
    - boost_system
    - boost_filesystem
    - jsoncpp
    - cublas

## Building SvComp
While SvComp was developed using Eclipse with the Nsight plugin, it's not a strict requirement. You can build it as long as you have the CUDA toolkit installed.

Here are the general steps to build SvComp:

- **Clone the Repository:**
 
```bash
git clone https://github.com/jjwilkerson/SvComp.git
cd SvComp
```

- **Building:**
If the above include paths, library paths, and libraries are configured then SvComp can be easily built (as an executable) in Eclipse with the Nsight plugin. It is necessary to set the PROGRAM_VERSION macro to a value of 1 or 2 to determine which executable program is built.

| PROGRAM_VERSION | Program                                             |
|:---------------:|:--------------------------------------------------- |
|        1        | Single training run                                 |
|        2        | Make vector index                                   |

Alternatively, you can build it from the command line using Make by executing one of the following commands. It may be necessary to update paths to nvcc and dependencies in Makefile first. You may also need to change the CUDA hardware versions after "arch=" to match your specific GPU.

Single training run:

```bash
make
```

Make vector index:

```bash
make index
```

Before building a different target, run the following:

```bash
make clean
```

## Usage
Before running the program, SED_DATASETS_DIR must be defined as an environment variable, with its value being the location of the files of vectors to be decoded and corresponding index files. The SentEncDec project can be run in encode mode to produce the vector files, and this project can be run in index mode to create the corresponding indexes. There should be one file of vectors for training, and another for testing. The files should have the following names:

- sentvec.out
- sentvec.out.ix
- sentvec_tune.out
- sentvec_tune.out.ix

To run as a single training run, place the executable in a new folder along with a config file named config.json. You can use the config.json file in the SvComp source folder as a starting point.

## Contributing
Contributions to SvComp are welcome. Please ensure to follow the coding standards and submit a pull request for review.

## License
SvComp is licensed under the MIT License. See the LICENSE file for more details.
