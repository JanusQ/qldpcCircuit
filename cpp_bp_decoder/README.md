# Belief Propagation Decoder for QLDPC Codes

A comprehensive C++ implementation of the Belief Propagation (BP) algorithm for decoding Quantum Low-Density Parity-Check (QLDPC) codes. This implementation solves the optimization problem:

```
s_L = argmax_{s_L} Σ_{e | D*e = s', D_L*e = s_L} exp(-Σ_j w_j * e_j)
```

where `w_j = ln((1-p_j)/p_j)` and `p_j` is the error probability for the j-th qubit.

## Algorithm Overview
To address your query for a detailed algorithm for Belief Propagation (BP) in the context of decoding QLDPC (Quantum Low-Density Parity-Check) codes, I'll provide a comprehensive, step-by-step procedure. The goal is to find the most likely logical syndrome \(\vec{s}_L\) given an observed syndrome \(\vec{s}'\), as per the optimization problem:

\[
\vec{s}_L = \arg\max_{\vec{s}_L} \sum_{\{\vec{e} \mid D \vec{e} = \vec{s}', D_L \vec{e} = \vec{s}_L\}} \exp\left(-\sum_j w_j e_j\right)
\]

where \(w_j = \ln\left(\frac{1-p_j}{p_j}\right)\), and \(p_j\) is the error probability for the \(j\)-th bit or qubit. BP is a message-passing algorithm that approximates the marginal probabilities needed to solve this problem by operating on a factor graph. Below is the detailed algorithm.

---

## **Detailed Algorithm for Belief Propagation (BP)**

### **Overview**
Belief Propagation is used here to approximate the joint probability \(P(\vec{s}', \vec{s}_L)\), which is proportional to the sum in the given expression. Since \(P(\vec{s}')\) is fixed for a given observed syndrome, maximizing \(P(\vec{s}', \vec{s}_L)\) over \(\vec{s}_L\) achieves the desired result. We construct a factor graph that includes variables for the error vector \(\vec{e}\) and the logical syndrome \(\vec{s}_L\), along with factors enforcing the constraints \(D \vec{e} = \vec{s}'\) and \(D_L \vec{e} = \vec{s}_L\). BP then computes approximate marginals for \(\vec{s}_L\).

---

### **Step 1: Construct the Factor Graph**
The factor graph is a bipartite graph with variable nodes and factor nodes, representing the variables and their constraints, respectively.

#### **Variable Nodes**
- **Error Variables**: \(e_1, e_2, \ldots, e_n\), where \(n\) is the number of physical bits or qubits, and each \(e_j \in \{0, 1\}\) represents a possible error.
- **Logical Syndrome Variables**: \(s_{L,1}, s_{L,2}, \ldots, s_{L,m_L}\), where \(m_L\) is the number of logical checks (rows of \(D_L\)), and each \(s_{L,b} \in \{0, 1\}\) is a component of \(\vec{s}_L\).

#### **Factor Nodes**
- **Syndrome Constraint Factors (\(f_a\))**: One for each of the \(m\) rows of the parity-check matrix \(D\), where \(m\) is the length of \(\vec{s}'\). For row \(a\):
  - **Connections**: Connected to \(e_j\) for all \(j\) where \(D_{a,j} = 1\) (denoted as the set \(N(a)\)).
  - **Function**: \(\delta\left(\bigoplus_{j \in N(a)} e_j, s'_a\right)\), where \(\bigoplus\) is the XOR operation (modulo 2 sum), \(s'_a\) is the \(a\)-th component of \(\vec{s}'\), and \(\delta(x, y) = 1\) if \(x = y\), else 0. This enforces \(D \vec{e} = \vec{s}'\).
- **Logical Constraint Factors (\(f_{L,b}\))**: One for each of the \(m_L\) rows of the logical check matrix \(D_L\). For row \(b\):
  - **Connections**: Connected to \(s_{L,b}\) and \(e_j\) for all \(j\) where \(D_{L,b,j} = 1\) (denoted as the set \(N_L(b)\)).
  - **Function**: \(\delta\left(\bigoplus_{j \in N_L(b)} e_j \oplus s_{L,b}, 0\right)\). This enforces \(s_{L,b} = \bigoplus_{j \in N_L(b)} e_j\), linking the error variables to the logical syndrome.
- **Prior Factors (\(f_j\))**: One for each \(e_j\).
  - **Connections**: Connected only to \(e_j\).
  - **Function**: \(P(e_j)\), where \(P(e_j = 0) = 1 - p_j\) and \(P(e_j = 1) = p_j\). Note that \(\exp(-w_j e_j) = (1 - p_j)^{1 - e_j} p_j^{e_j}\) up to normalization, so \(P(e_j) \propto \exp(-w_j e_j)\).

The \(s_{L,b}\) variables typically do not have prior factors unless specified, but a uniform prior (\(P(s_{L,b} = 0) = P(s_{L,b} = 1) = 0.5\)) can be assumed, though it doesn't affect the relative probabilities since we seek marginals.

---

### **Step 2: Initialize Messages**
Messages are functions or values passed between nodes, representing beliefs about variable states. For binary variables (\(e_j, s_{L,b} \in \{0, 1\}\)), each message is a pair of values (for 0 and 1).
- **Initial Messages**: Set all messages from variable nodes to factor nodes (\(\mu_{v \to f}\)) and from factor nodes to variable nodes (\(\mu_{f \to v}\)) to 1 for both possible values (i.e., \(\mu(0) = 1\), \(\mu(1) = 1\)). This is an uninformative initialization. Alternatively, small random values can be used to break symmetry.

---

### **Step 3: Iterate Belief Propagation Updates**
Use the sum-product algorithm to update messages iteratively until convergence or for a fixed number of iterations.

#### **Message Updates**
1. **Variable to Factor Message (\(\mu_{v \to f}(v)\))**:
   - For variable node \(v\) (either \(e_j\) or \(s_{L,b}\)) to factor node \(f\):
     \[
     \mu_{v \to f}(v) = P(v) \times \prod_{f' \in N(v) \setminus f} \mu_{f' \to v}(v)
     \]
     - If \(v = e_j\), \(P(e_j) = 1 - p_j\) for \(e_j = 0\), and \(p_j\) for \(e_j = 1\).
     - If \(v = s_{L,b}\), set \(P(s_{L,b}) = 1\) (effectively no prior), as the prior is uniform and factors out in the marginal.
     - \(N(v) \setminus f\) is the set of factor nodes connected to \(v\) excluding \(f\).

2. **Factor to Variable Message (\(\mu_{f \to v}(v)\))**:
   - For factor node \(f\) to variable node \(v\):
     \[
     \mu_{f \to v}(v) = \sum_{\sim v} \left[ f(\cdot) \prod_{u \in N(f) \setminus v} \mu_{u \to f}(u) \right]
     \]
     where \(\sum_{\sim v}\) means summing over all variables connected to \(f\) except \(v\), and \(f(\cdot)\) is the factor function.
   - **Prior Factor (\(f_j\))**: \(\mu_{f_j \to e_j}(e_j) = P(e_j)\).
   - **Syndrome Factor (\(f_a\))**: Connected to \(e_j\) for \(j \in N(a)\):
     \[
     \mu_{f_a \to e_k}(e_k) = \sum_{\{e_j \mid j \in N(a) \setminus k\}} \left[ \delta\left( \bigoplus_{j \in N(a)} e_j, s'_a \right) \prod_{j \in N(a) \setminus k} \mu_{e_j \to f_a}(e_j) \right]
     \]
     For binary variables, this can be computed efficiently:
     - Let \(q_j = \frac{\mu_{e_j \to f_a}(1)}{\mu_{e_j \to f_a}(0) + \mu_{e_j \to f_a}(1)}\) (probability \(e_j = 1\)).
     - Then:
       \[
       \mu_{f_a \to e_k}(0) = \frac{1}{2} \left( 1 + (1 - 2 s'_a) \prod_{j \in N(a) \setminus k} (1 - 2 q_j) \right)
       \]
       \[
       \mu_{f_a \to e_k}(1) = \frac{1}{2} \left( 1 - (1 - 2 s'_a) \prod_{j \in N(a) \setminus k} (1 - 2 q_j) \right)
       \]
   - **Logical Factor (\(f_{L,b}\))**: Connected to \(s_{L,b}\) and \(e_j\) for \(j \in N_L(b)\):
     - To \(e_k\):
       \[
       \mu_{f_{L,b} \to e_k}(e_k) = \sum_{s_{L,b}} \sum_{\{e_j \mid j \in N_L(b) \setminus k\}} \left[ \delta\left( s_{L,b} \oplus \bigoplus_{j \in N_L(b)} e_j, 0 \right) \mu_{s_{L,b} \to f_{L,b}}(s_{L,b}) \prod_{j \in N_L(b) \setminus k} \mu_{e_j \to f_{L,b}}(e_j) \right]
       \]
     - To \(s_{L,b}\):
       \[
       \mu_{f_{L,b} \to s_{L,b}}(s_{L,b}) = \sum_{\{e_j \mid j \in N_L(b)\}} \left[ \delta\left( s_{L,b} \oplus \bigoplus_{j \in N_L(b)} e_j, 0 \right) \prod_{j \in N_L(b)} \mu_{e_j \to f_{L,b}}(e_j) \right]
       \]
     Since the constraint is \(\bigoplus_{j \in N_L(b)} e_j \oplus s_{L,b} = 0\), this is a parity check with target 0, and similar efficient formulas apply:
     - For \(\mu_{f_{L,b} \to e_k}\), compute as above with \(s'_a = 0\).
     - For \(\mu_{f_{L,b} \to s_{L,b}}\), the message reflects the probability of \(s_{L,b}\) matching the parity of the \(e_j\)'s.

#### **Iteration**
- Repeat these updates for a fixed number of iterations (e.g., 10–50, depending on graph size) or until messages stabilize (e.g., maximum change in any message is below a threshold like \(10^{-6}\)).

---

### **Step 4: Compute Approximate Marginals for \(\vec{s}_L\)**
After convergence, calculate the belief (approximate marginal probability) for each \(s_{L,b}\):
- **Belief for \(s_{L,b}\)**:
  \[
  P(s_{L,b}) \propto \mu_{f_{L,b} \to s_{L,b}}(s_{L,b})
  \]
  Normalize so \(P(s_{L,b} = 0) + P(s_{L,b} = 1) = 1\).
- **Joint Probability for \(\vec{s}_L\)**:
  - If \(m_L\) is small, compute \(P(\vec{s}_L)\) for all \(2^{m_L}\) combinations by considering correlations (exact computation may require additional methods like junction trees, but BP approximates this).
  - Approximation: \(P(\vec{s}_L) \approx \prod_{b=1}^{m_L} P(s_{L,b})\), assuming independence, which holds if the logical checks are sparsely connected.

Since \(P(\vec{s}_L \mid \vec{s}') \propto P(\vec{s}', \vec{s}_L)\), and the factor graph enforces \(D \vec{e} = \vec{s}'\), the marginal \(P(\vec{s}_L)\) from BP is proportional to the desired sum.

---

### **Step 5: Select the Most Likely \(\vec{s}_L\)**
- For each **each possible \(\vec{s}_L\) (there are \(2^{m_L}\) possibilities), compute \(P(\vec{s}_L)\)** using the marginals from Step 4.
- Choose:
  \[
  \vec{s}_L = \arg\max_{\vec{s}_L} P(\vec{s}_L)
  \]
  Since \(m_L\) is typically small in QLDPC codes (e.g., 1 or 2 logical qubits), this enumeration is feasible.

---

## **Additional Notes**
- **Complexity**: Each iteration is \(O(n + m + m_L)\) per edge in the factor graph, making BP efficient for sparse QLDPC codes.
- **Accuracy**: BP is exact on tree-like graphs; for cyclic graphs (common in QLDPC), it’s approximate but often effective.
- **Numerical Stability**: Use log-domain computations (e.g., log-likelihood ratios) to avoid underflow/overflow.

## Implementation Features

- **Modular Design**: Clean separation between variable nodes, factor nodes, and the factor graph
- **Efficient Message Passing**: Optimized formulas for XOR constraints in binary fields
- **Multiple Examples**: Includes repetition codes and surface code-like structures
- **Convergence Detection**: Automatic stopping when messages stabilize
- **Marginal Computation**: Exact enumeration for small logical syndrome spaces
- **Memory Safe**: Modern C++17 with smart pointers and RAII

## File Structure

```
cpp_bp_decoder/
├── belief_propagation.hpp    # Header file with class definitions
├── belief_propagation.cpp    # Implementation of BP algorithm
├── example.cpp              # Comprehensive usage examples
├── Makefile                 # Build configuration
└── README.md               # This documentation
```

## Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Make (optional, for using the Makefile)

### Building

```bash
# Clone or download the files to cpp_bp_decoder/
cd cpp_bp_decoder

# Build the example
make

# Or compile manually
g++ -std=c++17 -O3 -Wall belief_propagation.cpp example.cpp -o bp_decoder_example
```

### Running

```bash
# Run the example program
make run

# Or run directly
./bp_decoder_example
```

## Usage Example

```cpp
#include "belief_propagation.hpp"
using namespace qldpc_bp;

// Define parity check matrix D and logical check matrix D_L
std::vector<std::vector<int>> D = {
    {1, 1, 0, 0, 0},
    {0, 1, 1, 0, 0},
    {0, 0, 1, 1, 0},
    {0, 0, 0, 1, 1}
};

std::vector<std::vector<int>> D_L = {
    {1, 1, 1, 1, 1}
};

// Set error probabilities
std::vector<double> error_probs = {0.1, 0.1, 0.1, 0.1, 0.1};

// Create decoder
BeliefPropagationDecoder decoder(D, D_L, error_probs);

// Set observed syndrome
std::vector<int> observed_syndrome = {1, 0, 0, 1};
decoder.set_observed_syndrome(observed_syndrome);

// Decode to find most likely logical syndrome
std::vector<int> logical_syndrome = decoder.decode(50, 1e-6);

// Get marginal probabilities
auto marginals = decoder.compute_logical_marginals();
```

## Algorithm Parameters

### Constructor Parameters

- **`D`**: Syndrome constraint matrix (m × n), where m is number of syndrome bits and n is number of physical qubits
- **`D_L`**: Logical constraint matrix (m_L × n), where m_L is number of logical syndrome bits
- **`error_probabilities`**: Vector of error probabilities for each physical qubit

### Decoding Parameters

- **`max_iterations`**: Maximum number of BP iterations (default: 50)
- **`convergence_threshold`**: Threshold for message convergence (default: 1e-6)

## Performance Characteristics

### Complexity
- **Time per iteration**: O(edges in factor graph) ≈ O(nnz(D) + nnz(D_L))
- **Space**: O(number of nodes + edges)
- **Total iterations**: Typically 10-50 for convergence

### Accuracy
- **Exact on trees**: BP is exact for tree-like factor graphs
- **Approximate on cycles**: Good approximation for sparse graphs with long cycles
- **Logical syndrome space**: Exact enumeration for ≤20 logical qubits, approximation for larger spaces

## Example Outputs

The example program demonstrates three scenarios:

### Example 1: Repetition Code
```
D matrix (syndrome constraints):
1 1 0 0 0 
0 1 1 0 0 
0 0 1 1 0 
0 0 0 1 1 

True error pattern: [1, 0, 0, 0, 1]
Observed syndrome: [1, 0, 0, 1]
True logical syndrome: [0]
Decoded logical syndrome: [0]
Decoding SUCCESSFUL
```

### Example 2: Surface Code-like Structure
```
9-qubit surface code with 6 stabilizers and 2 logical operators
Demonstrates handling of more complex constraint structures
```

### Example 3: Random Error Simulation
```
10 trials with random 15% error rate
Success rate tracking for statistical evaluation
```

## Advanced Features

### Custom Factor Graphs

You can extend the implementation by creating custom factor node types:

```cpp
class CustomFactorNode : public FactorNode {
public:
    Message compute_message_to_variable(int var_id) override {
        // Implement custom constraint logic
        return custom_message;
    }
};
```

### Message Schedules

The current implementation uses synchronous message passing. For better convergence, you can implement:
- Flooding schedule (current)
- Sequential schedule  
- Residual belief propagation

### Numerical Stability

For better numerical stability with very small probabilities:
- Use log-domain arithmetic
- Implement log-likelihood ratios (LLRs)
- Add numerical safeguards for edge cases

## Theoretical Background

### QLDPC Decoding Problem

QLDPC codes are quantum error-correcting codes with sparse parity-check matrices. The decoding problem involves:

1. **Physical Syndrome**: Measured syndrome `s'` from stabilizer measurements
2. **Logical Syndrome**: Unknown logical syndrome `s_L` to be determined
3. **Error Model**: Each qubit has independent error probability `p_j`

### Factor Graph Representation

The joint probability distribution factors as:
```
P(e, s_L | s') ∝ ∏_j P(e_j) × ∏_a δ(D_a·e, s'_a) × ∏_b δ(D_L,b·e ⊕ s_L,b, 0)
```

This factorization directly corresponds to the factor graph structure used in the implementation.

## References

1. **QLDPC Codes**: Quantum Low-Density Parity-Check codes and their properties
2. **Belief Propagation**: Pearl, J. "Probabilistic Reasoning in Intelligent Systems" (1988)
3. **Factor Graphs**: Kschischang, F.R., Frey, B.J., Loeliger, H.A. "Factor graphs and the sum-product algorithm" (2001)
4. **LDPC Decoding**: MacKay, D.J.C. "Information Theory, Inference and Learning Algorithms" (2003)

## Contributing

This implementation provides a solid foundation for QLDPC decoding research. Potential improvements include:

- Generalized belief propagation for higher-order constraints
- Optimized numerical implementations
- Integration with quantum error correction frameworks
- Benchmarking against other decoding algorithms

## License

This implementation is provided for educational and research purposes. Please cite appropriately if used in academic work. 