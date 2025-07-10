# Mean-Field QLDPC Decoder Implementation

This directory contains a complete implementation of the mean-field decoder for CSS quantum LDPC codes, as described in the theoretical README. The implementation provides both C++ and Python interfaces for efficient decoding.

## Overview

The mean-field decoder solves the following equations iteratively:

```
n_i = tanh(sum_j (w_j/2) * (-1)^e_{0j} * (L_x)_{ji} * prod_k m_k^{(D_x)_{jk}} * prod_{i'!=i} n_{i'}^{(L_x)_{ji'}})
m_k = tanh(sum_j (w_j/2) * (-1)^e_{0j} * (D_x)_{jk} * prod_{k'} m_{k'}^{(D_x)_{jk'}} * prod_i n_i^{(L_x)_{ji}})
```

where:
- `w_j = ln((1-p_j)/p_j)` are the error weights
- `e_{0j}` are the initial error syndrome bits
- `D_x` and `L_x` are sparse matrices in CSC format
- `n_i` are the logical spin magnetizations
- `m_k` are the check spin magnetizations

## Files

- `meanfield_decoder.cpp` - Core C++ implementation
- `pybind_wrapper.cpp` - Python binding using pybind11
- `qlpc_decoder.py` - High-level Python interface
- `setup.py` - Build configuration
- `example.py` - Usage examples
- `test_decoder.py` - Unit tests
- `requirements.txt` - Python dependencies
- `Makefile` - Build automation

## Installation

### Prerequisites

- Python 3.7 or higher
- C++ compiler with C++17 support
- pybind11
- numpy, scipy

### Quick Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build the extension:
```bash
make build
```

3. Install in development mode:
```bash
make install-dev
```

### Manual Installation

```bash
# Install dependencies
pip install numpy scipy pybind11

# Build and install
python setup.py build_ext --inplace
pip install -e .
```

## Usage

### Basic Usage

```python
import numpy as np
import scipy.sparse as sp
from qlpc_decoder import QLDPCDecoder, create_uniform_error_model

# Create a simple CSS code
D_x_data = [
    [1, 1, 0, 0, 0, 0],  # Check 1: qubits 0,1
    [0, 1, 1, 0, 0, 0],  # Check 2: qubits 1,2
    [0, 0, 0, 1, 1, 1],  # Check 3: qubits 3,4,5
]
L_x_data = [
    [1, 0, 0, 1, 0, 0],  # Logical 1: qubits 0,3
    [0, 1, 0, 0, 1, 0],  # Logical 2: qubits 1,4
]

D_x = sp.csr_matrix(D_x_data, dtype=float)
L_x = sp.csr_matrix(L_x_data, dtype=float)

# Create decoder
decoder = QLDPCDecoder(max_iterations=100, tolerance=1e-6)
decoder.set_code(D_x, L_x)

# Create error model
error_probs = create_uniform_error_model(6, 0.1)  # 10% error rate

# Generate syndrome
error_pattern = np.array([1, 0, 0, 0, 1, 0])
syndrome = (D_x @ error_pattern) % 2

# Decode
logical_correction = decoder.decode_syndrome(syndrome, error_probs)
print(f"Logical correction: {logical_correction}")
```

### Advanced Usage

```python
# Batch decoding
syndromes = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])
logical_corrections = decoder.decode_batch(syndromes, error_probs)

# Get magnetizations for analysis
check_mags, logical_mags = decoder.get_magnetizations(syndrome, error_probs)

# Custom error model
weights = np.log((1 - error_probs) / error_probs)
logical_correction = decoder.decode_syndrome(syndrome, weights=weights)
```

### Error Models

```python
from qlpc_decoder import create_uniform_error_model, create_depolarizing_error_model

# Uniform error model
error_probs = create_uniform_error_model(num_qubits=10, error_prob=0.05)

# Depolarizing error model
error_probs = create_depolarizing_error_model(
    num_qubits=10, 
    p_x=0.02,  # X error probability
    p_y=0.01,  # Y error probability
    p_z=0.02   # Z error probability
)
```

## API Reference

### QLDPCDecoder Class

#### Constructor
```python
QLDPCDecoder(max_iterations=100, tolerance=1e-6)
```

#### Methods

- `set_code(D_x, L_x)` - Set the CSS code parameters
- `decode_syndrome(syndrome, error_probs=None, weights=None)` - Decode a single syndrome
- `decode_batch(syndromes, error_probs=None, weights=None)` - Decode multiple syndromes
- `get_magnetizations(syndrome, error_probs=None, weights=None)` - Get final magnetizations
- `compute_eta(syndrome)` - Compute eta values
- `set_parameters(max_iterations, tolerance)` - Update convergence parameters
- `error_probabilities_to_weights(error_probs)` - Convert probabilities to weights

#### Properties

- `code_info` - Information about the current code

### Utility Functions

- `create_uniform_error_model(num_qubits, error_prob)` - Create uniform error model
- `create_depolarizing_error_model(num_qubits, p_x, p_y, p_z)` - Create depolarizing error model
- `decode_css_code(D_x, L_x, syndrome, error_probs, ...)` - One-shot decoding function

## Building from Source

### Development Build

```bash
# Clean and build
make clean
make build

# Run tests
make test

# Run example
make example
```

### Production Build

```bash
# Install with optimizations
python setup.py build_ext --inplace --compiler=unix
pip install -e .
```

### Custom Compilation

```bash
# Manual compilation with custom flags
python setup.py build_ext --inplace \
    --compiler=unix \
    --compiler-options="-O3 -march=native -ffast-math"
```

## Performance

The C++ implementation is optimized for performance:

- **Sparse Matrix Operations**: Uses CSC format for efficient sparse matrix operations
- **Memory Efficiency**: Minimal memory allocation during iterations
- **SIMD Optimizations**: Compiler optimizations for vectorized operations
- **Convergence**: Fast convergence with configurable tolerance

Typical performance:
- Small codes (< 100 qubits): ~1ms per syndrome
- Medium codes (100-1000 qubits): ~10ms per syndrome
- Large codes (> 1000 qubits): ~100ms per syndrome

## Testing

Run the test suite:

```bash
# Run all tests
make test

# Run specific test file
python -m pytest test_decoder.py -v

# Run with coverage
python -m pytest test_decoder.py --cov=qlpc_decoder
```

## Examples

See `example.py` for comprehensive examples including:

- Single syndrome decoding
- Batch decoding
- Convergence analysis
- Error threshold analysis
- Visualization of results

## Troubleshooting

### Build Issues

1. **pybind11 not found**: Install with `pip install pybind11`
2. **C++17 not supported**: Update your compiler or use `--std=c++17`
3. **numpy headers not found**: Install numpy development headers

### Runtime Issues

1. **Import error**: Ensure the extension is built and installed
2. **Memory error**: Reduce matrix size or increase system memory
3. **Convergence issues**: Increase `max_iterations` or adjust `tolerance`

### Performance Issues

1. **Slow convergence**: Try different initial conditions or adjust parameters
2. **Memory usage**: Use sparse matrices and monitor memory usage
3. **Compilation**: Ensure optimizations are enabled (`-O3`)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite
5. Submit a pull request

## License

This implementation is provided under the MIT License. See the main project license for details.

## References

- Mean-field theory for spin glasses
- CSS quantum error correction codes
- Belief propagation algorithms
- Sparse matrix operations 