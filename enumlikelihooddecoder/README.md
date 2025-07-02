# QLDPCDecoder Python Extension

This package provides a Python interface to the QLDPCDecoder C++ class using pybind11.

## Build Instructions

1. Install pybind11 and CMake:
   ```bash
   pip install pybind11
   sudo apt install cmake
   ```

2. Build the extension:
   ```bash
   cd maxlikelihooddecoder
   mkdir build
   cd build
   cmake ..
   make
   ```
   This will produce a `maxlikelihooddecoder.*.so` file (the Python extension module).

3. (Optional) Move the `.so` file to your Python package directory or add the build directory to your `PYTHONPATH`.

## Python Usage Example

```python
import maxlikelihooddecoder

D_prime = [
    [1, 0, 0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 0]
]
priors = [0.01] * 7

decoder = maxlikelihooddecoder.QLDPCDecoder(D_prime, priors, 2, 1)
decoder.print_stats()
syndrome_prime = [0, 0, 1]
decoded = decoder.decode(syndrome_prime)
print("Decoded:", decoded)
``` 