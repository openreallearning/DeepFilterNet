# C++ Voice Extraction Example

This folder contains a minimal example showing how to load the
DeepFilterNet model weights directly with ONNX Runtime and process a noisy
audio file in C++.

The example depends on `libsndfile` for reading/writing WAV files and on
`onnxruntime` for running the neural networks. Both libraries are
included as git submodules in the repository. After cloning the project,
initialise the submodules with:

```bash
git submodule update --init --recursive
```

Build ONNX Runtime and libsndfile from the submodules and point the compiler
at their generated headers and libraries.

## Build

1. Compile the example manually (adjust the paths to the generated headers
   and libraries if necessary)
   ```bash
   g++ enhance.cpp -std=c++17 -I../onnxruntime/include -I../libsndfile/include \
       -L../libsndfile/lib -L../onnxruntime/build/Linux/Release \
       -lsndfile -lonnxruntime -o enhance
   ```

2. Alternatively, build with CMake
   ```bash
   cmake -S . -B build
   cmake --build build
   ```
   This will also build libsndfile from the submodule.

## Usage

```
./enhance <model.tar.gz> <input.wav> <output.wav>
```

The model archive is extracted automatically into a temporary directory.
