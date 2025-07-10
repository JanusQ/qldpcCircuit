# Circuit level Quantum Error Correction for QLDPC Codes

# maximum likelihood decoder design
## Formulation

### Problem definition
The decoding of QLDPC code has the following relationship

$$
\begin{bmatrix}
D \\ D_L
\end{bmatrix}\vec{e} = \begin{bmatrix}
\vec{s'} \\ \vec{s}_L
\end{bmatrix}
$$
where $D$ is decoding matrix, and $D_L$ is logical check matrix, $s'$ is the syndrome, and $\vec{s}_L$ is the logical syndrome. The error vector $\vec{e}$ has a prior probability $p_j$ for every $e_j\in\{0,1\}$. We can note this relationship as $D'\vec{e} = \vec{s'}$.
Here, all matrix-vector multiplication are all mod 2.
Here we want to find the most likely $\vec{s}_L$ for given $\vec{s'}$.
$$\vec{s}_L = \text{argmax}_{\vec{s_l}} \sum_{\{\vec{e}|D'\vec{e} = \vec{s}\}} e^{-\sum_j w_j e_j}$$
here, $w_j = ln((1-p_j)/p_j)$ is the weight of each bit.

### enumeration approach
The lower hamming weights in $\vec{e}$, the lower probability it occurs. We aim to enumerate the low hamming weights error vector. 
For each error vector, we can use the 
$$\begin{bmatrix}
D \\ D_L
\end{bmatrix}\vec{e} = \begin{bmatrix}
\vec{s'} \\ \vec{s}_L
\end{bmatrix}$$
to get the $\vec{s'}, \vec{s}_L$,  and we summeries the  $e^{-\sum_j w_j e_j}$ with same  $\vec{s'}, \vec{s}_L$ as likelihood $Z(\vec{s'}, \vec{s}_L)$. Then for every input $\vec{s'}$, we find the $\vec{s}_L$ that maxmize the $Z(\vec{s'}, \vec{s}_L)$, that is the decoded logical syndrome. 

To test the logical syndrome rate, we randomly sample the error by the prior probability, and log the syndrome and true logical syndrome by 
$$
\begin{bmatrix}
D \\ D_L
\end{bmatrix}\vec{e} = \begin{bmatrix}
\vec{s'} \\ \vec{s}_L
\end{bmatrix}
$$
Then we seed the syndrome to the decoder and get the decoded logical syndrome. If the decoding matches with the true logical syndrome, the decoding success. the logical error rate is the success rate of decoding. 

## Eliminate approach

When $D' = [I \mid B]$ and $\vec{e} = [\vec{e}_1^T \mid \vec{e}_2^T]^T$, the constraint is:
$$\vec{e}_1 \oplus B\vec{e}_2 = \vec{s}$$

For binary variables, this means:
$$\vec{e}_1 = \vec{s} \oplus B\vec{e}_2$$

The energy term becomes:
$$\sum_j w_j e_j = \vec{w}_1^T \vec{e}_1 + \vec{w}_2^T \vec{e}_2 = \vec{w}_1^T (\vec{s} \oplus B\vec{e}_2) + \vec{w}_2^T \vec{e}_2$$

Since $\vec{s} \oplus B\vec{e}_2 = \vec{s} + B\vec{e}_2 - 2(\vec{s} \odot B\vec{e}_2)$ (where $\odot$ is element-wise multiplication), we get:

$$\vec{w}_1^T (\vec{s} \oplus B\vec{e}_2) = \vec{w}_1^T \vec{s} + \vec{w}_1^T B\vec{e}_2 - 2\vec{w}_1^T (\vec{s} \odot B\vec{e}_2)$$

## New Formulation

The optimization becomes:
$$\vec{s}^* = \arg\min_{\vec{s}} \sum_{\vec{e}_2} \exp\left(-\vec{w}_1^T \vec{s} - \vec{w}_1^T B\vec{e}_2 + 2\vec{w}_1^T (\vec{s} \odot B\vec{e}_2) - \vec{w}_2^T \vec{e}_2\right)$$

This can be rewritten as:
$$\vec{s}^* = \arg\min_{\vec{s}} e^{-\vec{w}_1^T \vec{s}} \sum_{\vec{e}_2} \exp\left(-\vec{w}_1^T B\vec{e}_2 - \vec{w}_2^T \vec{e}_2 + 2\vec{w}_1^T (\vec{s} \odot B\vec{e}_2)\right)$$

$$\ln \vec{s_L}^* = \arg\min_{\vec{s}=\begin{bmatrix}
\vec{s'} \\ \vec{s}_L
\end{bmatrix}} -\vec{w}_1^T \vec{s}+ \ln \sum_{\vec{e}_2} \exp\left(-\vec{w}_1^T B\vec{e}_2 - \vec{w}_2^T \vec{e}_2 + 2\vec{w}_1^T (\vec{s} \odot B\vec{e}_2)\right)$$
Here, $\vec{s'}$ is the syndrome. and we want find $\vec{s_L}$.

## Key Insight
The lower hamming weights in $\vec{e}$, the lower probability it occurs. We aim to enumerate the low hamming weights error vector. 

Write a cpp implementation of the above algorithm.
You should also write a test function to test the logical syndrome rate by repeat generation error using the prior probability and then using the decoding equation to get the syndrome and the logical syndrome,  if the decoded logical syndrome matches the original logical syndrome, the decoding success. The logical syndrome rate is the success times/ all repeat times.

###

I want to minimize $\vec{l}^* = argmin_{\vec{l}\in{0,1}^K} \sum_{\vec{u} \in {0,1}^m} P(\vec{e_0}\oplus D_x \vec{u} \oplus L_x \vec{l})$, where D_x is the X-type check matrix, L_x is the X logical operators, and the matrix-vector product is module 2.  \vec{e_0} is a know initial X-error. P(\vec{e}) is the probability of error $\vec{e}$ and can be expressed as $P(\vec{e})=e^{- \sum_{j} w_j e_j}$ ($w_j = ln((1-p_j)/p_j)$) or  $P(\vec{e})=\prod_{j} (1+ (z_j-1)e_j)$($z_j= p_j/(1-p_j)$). Is there any polynominal way to simplify this expresion or caculate the probability summation in polynomical way?