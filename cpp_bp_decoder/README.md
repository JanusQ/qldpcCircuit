# Belief Propagation with accumlated likelihood Decoder for QLDPC Codes

The decoding of QLDPC code has the following relationship

$$
\begin{bmatrix}
D \\ D_L
\end{bmatrix}\vec{e} = \begin{bmatrix}
\vec{s'} \\ \vec{s}_L
\end{bmatrix}
$$
where $D$ is decoding matrix, and $D_L$ is logical check matrix, $s'$ is the syndrome, and $\vec{s}_L$ is the logical syndrome. The error vector $\vec{e}$ has a prior probability $p_j$ for every $e_j\in\{0,1\}$. We can note this relationship as $D'\vec{e} = \vec{s'}$.

Here we want to find the most likely $\vec{s}_L$ for given $\vec{s'}$.
$$\vec{s}_L = \text{argmax}_{\vec{s_l}} \sum_{\{\vec{e}|D'\vec{e} = \vec{s}\}} e^{-\sum_j w_j e_j}$$
here, $w_j = ln((1-p_j)/p_j)$ is the weight of each bit.

However, this problem is #P-complete, which make exact decoding impossible. We observed that if the error has low hamming weight, the probability will be larger. So counting the low-weight error is a good approximation for the summation.
However, enumeration all low-weight error is still expensive. 
Belief propagation aims to find marginal probabality of each qubit error to satisfy the check syndrome. Our instution is to leverage iterations of BP as a sampler to the error vector. we accumulated all valid error to find the most like $S_L$.

## Installation

### Requirements
- C++ compiler with C++11 support
- Python >= 3.6
- numpy >= 1.19.0
- pybind11 >= 2.6.0

### Building from source

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build the C++ extension:
```bash
make build
# or
python setup.py build_ext --inplace
```

3. (Optional) Install the package:
```bash
make install
# or
pip install -e .
```

### Testing
Run the test suite to verify the installation:
```bash
make test
# or
python test_decoder.py
```

For logical error rate analysis:
```bash
make test-logical
# or
python test_logical_error_rate.py
```

For scaling analysis with multiple code sizes:
```bash
make test-scaling
# or
python test_scaling_analysis.py
```

## Python Usage Example

```python
from qldpcdecoding.codes import gen_BB_code, gen_HP_ring_code, get_benchmark_code
import numpy as np
from qldpcdecoding.utils import gauss_elimination_mod2
from BPdecoder import likelihoodDecoder

css_code = gen_BB_code(72)
D = css_code.hz
DL = css_code.lz

# Create decoder with default name
decoder = likelihoodDecoder(D, DL, [0.001] * D.shape[1], 3)
print(f"Decoder name: {decoder.name}")  # Output: BP_AccumulatedLikelihood

# Create decoder with custom name
decoder_v2 = likelihoodDecoder(D, DL, [0.001] * D.shape[1], 3, name="MyDecoder")
print(f"Custom decoder name: {decoder_v2.name}")  # Output: MyDecoder

# Decode
error = np.random.choice(2, D.shape[1], replace=True, p=[0.001, 0.999])
syndrome = np.mod(D @ error, 2)
decoded_logic = decoder.decode(syndrome)
print("Decoded:", decoded_logic)
assert np.allclose((DL @ error)%2, decoded_logic)
```

## Implementation Details

The C++ implementation includes:
- **Sparse matrix representation** for efficient LDPC code handling
- **Belief Propagation algorithm** with log-likelihood ratio (LLR) messages
- **Accumulated likelihood tracking** for different logical syndromes
- **Sampling-based approach** using BP marginals to generate error samples
- **Python bindings** via pybind11 for seamless integration

The decoder uses the BP iterations as a sampler to generate error vectors consistent with the observed syndrome, accumulating their likelihoods to determine the most probable logical syndrome.

## API Reference

### likelihoodDecoder

Constructor:
```python
likelihoodDecoder(D, DL, error_probs, iterations, name="BP_AccumulatedLikelihood")
```

Parameters:
- `D`: Decoding matrix (numpy array, dtype=int32)
- `DL`: Logical check matrix (numpy array, dtype=int32)
- `error_probs`: Physical error probabilities for each qubit (list or numpy array, dtype=float64)
- `iterations`: Number of BP iterations to run (int)
- `name`: Optional decoder name (string, default="BP_AccumulatedLikelihood")

Properties:
- `name`: Read-only property returning the decoder name

Methods:
- `decode(syndrome)`: Decode the syndrome and return the most likely logical syndrome 