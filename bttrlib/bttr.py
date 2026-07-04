import warnings

import numpy as np
import scipy
import tensorly as tl
from tensorly import decomposition, tenalg

from .ace import optimize_tensor_decomposition


def fix_numpy_vector(vector):
    r"""
    Hack to properly encode 1D arrays in Python

    Args:
        vector (np.array): the vector to properly encode
    """
    return vector[None].T if len(vector.shape) == 1 else vector

class BTTR:
    r"""
    Block-Term Tensor Regression (BTTR)
    """

    def train(self, X : np.ndarray, Y : np.ndarray, nFactor : int, SNRs=range(30,41), ratios=np.arange(95,99.9, 1), useACCoS=False, use_deflate=(True, True), enforce_rank_1=False, score_vector_matrix=False):
        r"""
        Block-Term Tensor Regression (BTTR)

        BTTR is a deflation-based method in which the maximally correlated representations of :math:`\underline{\mathbf{X}}` and :math:`\mathbf{y}` are extracted via ACE/ACCoS at each iteration. Therefore, BTTR inherits the advantages of the proposed ACE/ACCoS and does not require one to set the model parameters manually. This provides BTTR with an additional important property: the ability to model complex data in which the optimal MTR is not necessarily stable across sequential decompositions.

        Given a set of training data :math:`\underline{\mathbf{X}}_{\text{train}} \in \mathbb{R}^{I_1 \times ... \times I_N}` and vectoral response :math:`\mathbf{y}_{\text{train}} \in  \mathbb{R}^{I_1}`, BTTR training consists of automatically identifying K blocks s.t.

        .. math::
            \underline{\mathbf{X}}_{\text{train}} = \sum^K_{k=1} \underline{\mathbf{G}}_{k} \times_1 \mathbf{t}_k \times_2 \mathbf{P}_k^{(2)} \times_3 ... \times_N \mathbf{P}_k^{(n)} + \underline{\mathbf{E}}_{k} \\
            \mathbf{y}_{\text{train}} = \sum^K_{k=1} \mathbf{u}_k + \mathbf{f}_k \text{ with } \mathbf{u}_k = \mathbf{t}_k b_k

        with :math:`\underline{\mathbf{G}}_{k} \in \mathbb{R}^{1 \times R_2^k \times ... \times R_N^k}` the core tensor for the kth-block, :math:`\mathbf{P}_k^{(n)}` the kth loading matrix for the n-mode, :math:`\mathbf{u}_k` and :math:`\mathbf{t}_k` the score vectors, :math:`b_k` the regression coefficient, and :math:`\underline{\mathbf{E}}_{k}` and :math:`\mathbf{f}_k` the residuals. Once the model is trained - and, hence, :math:`\underline{\mathbf{G}}_{k}`, :math:`\mathbf{P}_k^{(n)}` and :math:`b_k` are computed - the final prediction is obtained as: :math:`\mathbf{y}_{\text{test}} = \mathbf{Tb} = \mathbf{X}_{\text{test(1)}}\mathbf{Wb}` where each column :math:`\mathbf{w}_k = (\mathbf{P}_k^{(n)} \otimes ... \otimes \mathbf{P}_k^{(2)}) vec(\underline{\mathbf{G}}_{k})`.

        .. pcode::
            :linenos:

            \begin{algorithm}
            \begin{algorithmic}
            \REQUIRE $\underline{\mathbf{X}} \in \mathbb{R}^{I_1 \times ... \times I_N}, \mathbf{y} \in \mathbb{R}^{I_1 \times 1}, K$
            \ENSURE $\{ \mathbf{P}_k^{(n)} \}, \{ \mathbf{t}_k \}, \underline{\mathbf{G}}_k^{(X)}$ for $k=1,...,K$ ; $n=2,...,N$

            \STATE \textbf{Initialisation of} $\underline{\mathbf{E_1}} = \underline{\mathbf{X}}$ and $\mathbf{f_1} = \mathbf{y}$

            \FOR {$k=1$ to $K$}
                \IF{$\lVert\underline{\mathbf{E_k}}\lVert > \epsilon$ and $\lVert\mathbf{f_k}\lVert > \epsilon$}
                    \STATE $\underline{\mathbf{G}}_k^{(X)}, \mathbf{t}_k, \mathbf{P}_k^{(2)}, ..., \mathbf{P}_k^{(N)} $ = \textit{ACE}($\underline{\mathbf{E}}_k, \mathbf{f}_k$) or \textit{ACCoS}($\underline{\mathbf{E}}_k, \mathbf{f}_k$)

                    \STATE $\mathbf{b_k} = \mathbf{t_k} \mathbf{f_k}$
                    \STATE $\underline{\mathbf{E_{k+1}}} = \underline{\mathbf{E_{k}}} - \llbracket \underline{\mathbf{G}}_k^{(X)} ; \mathbf{t}_k, \mathbf{P}_k^{(2)},...,\mathbf{P}_k^{(N)} \rrbracket$
                    \STATE $\mathbf{f_{k+1}} = \mathbf{f_{k}} - \mathbf{t_k} \mathbf{b_k}$
                \ELSE
                    \STATE \textbf{break}
                \ENDIF
            \ENDFOR
            \end{algorithmic} 
            \end{algorithm}

        Args:
            X (np.ndarray): Your input tensor (as a multidimensional numpy array)
            Y (np.ndarray): The output (as a multidimensional numpy array) you want to model. 
                            This output can be 1-dimensional (BTTR), 2-dimensional (Extended BTTR) or higher-dimensional (Generalised BTTR)
            nFactor: The number of blocks that BTTR needs to make
            SNRs (float/int list): list of parameters for use in the updating of the core tensor
            ratios (float/int list): list of parameters for use in pruning of the components
            useACCoS (bool): Used to select whether or not ACCoS should be used 
            use_deflate ((bool, bool) tuple): Used to set whether (X, Y) should be deflated during each iteration
            enforce_rank_1 (bool): Set whether Rank-(L2, . . . , LN , K2, . . . , KM) decomposition (for general purpose) should be used or Rank-(L2, · · · , LN, 1) decomposition (for tensor-matrix or tensor-vector only) should be used
            score_vector_matrix (bool): Used to set whether the score vector should be created differently (as described in eBTTR paper, only tensor-matrix or tensor-vector)
        """
        if score_vector_matrix: enforce_rank_1 = True

        self.E = BlockTermTensor(X, pseudo_inverse=True, score_vector_matrix=score_vector_matrix)
        self.F = BlockTermTensor(Y, score_vector_matrix=score_vector_matrix)
        self.nFactor = nFactor
        self.ace_results = []

        tolerance = 1e-16

        for f in range(0, nFactor):
            if self.E.norm() > tolerance and self.F.norm() > tolerance:
                decomp = Decomposition(self.E, self.F, enforce_rank_1)
                res = decomp.optimize_tensor_decomposition(SNRs, ratios, useACCoS)
                self.ace_results.append(res)
                decomp.deflate(use_deflate, score_vector_matrix)
            else:
                break
    
    def predict(self, Xtest):
        r"""
        Once the model is trained – and, hence, :math:`\underline{\mathbf{G}}_{k}`, :math:`\mathbf{P}_k^{(n)}` and :math:`b_k` are computed – the final prediction is obtained as: :math:`\mathbf{y}_{\text{test}} = \mathbf{Tb} = \mathbf{X}_{\text{test(1)}}\mathbf{Wb}` where each column :math:`\mathbf{w}_k = (\mathbf{P}_k^{(n)} \otimes ... \otimes \mathbf{P}_k^{(2)}) vec(\underline{\mathbf{G}}^+_{k})`.

        Args:
            Xtest (np.ndarray): Input tensor (as a multidimensional numpy array). If the tensor is only comprised of a single sample, the dimension needs to be 1 (sample) x ....
        """
        if not self.nFactor: return [None]
        out = [None] * self.nFactor
        for i in range(0, self.nFactor):
            out[i] = Xtest.reshape((Xtest.shape[0], -1), order='F') @ (self.E.model(i+1) @ self.F.model(i+1))
        return out
    
    def getModel(self, factor=None):
        return self.E.model(factor), self.F.model(factor)

class BlockTermTensor:
    r"""
    BlockTermTensor defines the basic block for a Tensor within Block-Term Tensor Regression
    """
    
    def __init__(self, tensor, pseudo_inverse=False, score_vector_matrix=False):
        r"""
        BlockTermTensor defines the basic block for a Tensor within Block-Term Tensor Regression

        Args:
            tensor (np.ndarray): Tensor to encapsulate within the BlockTermTensor
        """
        self.tensor = fix_numpy_vector(tensor)
        self.function = lambda x: np.linalg.pinv(x).T if pseudo_inverse else x
        self.transpose = pseudo_inverse
        self.score_vector_matrix = score_vector_matrix
        self.models = []

    def deflate(self, score_vector_t, components, deflate=True, core_tensor_G=None):
        r"""
        Deflate function

        Args:
            score_vector_t (np.array): Score vector necessary to reconstruct the tensor
            components (np.array list): Factor matrices necessary to reconstruct the tensor
            deflate (bool): Used to set whether the tensor should be deflated during each iteration
        """
        core_tensor = tl.tucker_to_tensor((self.tensor, [score_vector_t] + components), transpose_factors=True)
        if deflate: self.tensor = self.tensor - tl.tucker_to_tensor((core_tensor, [score_vector_t] + components))
        """ Precompute the 'model' used to predict in the predict function """
        if (not self.score_vector_matrix) or (core_tensor_G is None): 
            self.models.append(tenalg.kronecker(components, reverse=True) @ self.function(fix_numpy_vector(core_tensor.reshape((-1), order='F'))))  # type: ignore
        else:
            """
                In the Matlab code the following is used
                This forms the exact same result
                This doesn't follow the math as set up in the HOPLS paper 
                [Higher-Order Partial Least Squares (HOPLS): A Generalized Multi-Linear Regression Method]
                Only works for rank_1 decomposition (hence the self.score_vector_matrix check)
            """
            wpls = tenalg.kronecker(components, reverse=True) @ fix_numpy_vector(core_tensor_G.reshape((-1), order='F')) # type: ignore
            ppls = np.linalg.inv((core_tensor.reshape((-1), order='F') @ fix_numpy_vector(core_tensor_G.reshape((-1), order='F')))[None]) # type: ignore
            self.models.append(wpls @ ppls)

    def model(self, factor=None):
        r"""
        Return the prediction matrix
        """
        if not factor: factor = len(self.models)
        W = tl.tensor([self.models[f] for f in range(0, factor)])
        W = W.reshape(W.shape[:-1])
        if self.transpose: W = W.T
        return W
    
    def norm(self):
        r"""
        Returns a tensor norm
        """
        return np.linalg.norm(self.tensor)

class Decomposition:
    r"""
    .. pcode::
        :linenos:

        \begin{algorithm}
        \begin{algorithmic}
        \REQUIRE $\underline{\mathbf{X}} \in \mathbb{R}^{I_1 \times ... \times I_N}, \mathbf{y} \in \mathbb{R}^{I_1 \times 1}$
        \ENSURE $\underline{\mathbf{G}}^{(X)} \in \mathbb{R}^{1 \times R_2 \times ... \times R_N}, \mathbf{t}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$

        \STATE $\underline{\mathbf{G}}, \mathbf{t}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$ = \textit{ACE}($\underline{\mathbf{X}}, \mathbf{y}$)

        \IF{\textit(ACCoS) is enabled}
            \STATE \textbf{select} correlated components using \textit{ACCoS}
        \ENDIF

        \STATE $\mathbf{t} = (\underline{\mathbf{X}} \times_2 \mathbf{P}^{(2)T} \times_3 ... \times_N \mathbf{P}^{(N)T})_{(1)}\textit{vec}(\underline{\mathbf{G}})$
        \STATE $\mathbf{t} = \mathbf{t} / \lVert\mathbf{t}\lVert_F$
        \STATE $\underline{\mathbf{G}}^{(X)} = \llbracket \underline{\mathbf{X}} ; \mathbf{t}^T, \mathbf{P}^{(2)T},...,\mathbf{P}^{(N)T} \rrbracket$

        \RETURN $\underline{\mathbf{G}}^{(X)}, \mathbf{t}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$ 
        \end{algorithmic} 
        \end{algorithm}
    """

    def __init__(self, X, Y, enforce_rank_1=False):
        r"""
        Wrapper class around an iteration of BTTR

        During initialisation, a mode-1 cross-covariance tensor is created (read: python is zero-indexed and thus mode-1 is encoded as 0). Afterwards, the rank is set and the mode-1 cross-covariance tensor is Tucker decomposed.

        Args:
            X (BlockTermTensor): Main Input Tensor
            X (BlockTermTensor): Second Input Tensor
            enforce_rank_1 (bool): Set whether Rank-(L2, . . . , LN , K2, . . . , KM) decomposition (for general purpose) should be used or Rank-(L2, · · · , LN, 1) decomposition (for tensor-matrix or tensor-vector only) should be used
        """
        self.X, self.Y = X, Y
        self.full_tensor_C = np.array(np.tensordot(self.X.tensor, self.Y.tensor, (0, 0)))
        rank = np.array(list(self.full_tensor_C.shape)) # Rank-(L2, . . . , LN , K2, . . . , KM)
        if enforce_rank_1: rank[len(self.X.tensor.shape)-1:] = 1 # Rank-(L2, · · · , LN, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # compat: tensorly>=0.8 validate_tucker_rank rejects ndarray rank -> pass a list
            self.core_tensor_G, self.components = decomposition.tucker(self.full_tensor_C, rank.tolist())

    def optimize_tensor_decomposition(self, SNRs=range(1,50), ratios=np.arange(95,99.9, 0.1), useACCoS=False):
        r"""
        Optimization and automatic parameter estimation

        Args:
            SNRs (float/int list): list of parameters for use in the updating of the core tensor
            ratios (float/int list): list of parameters for use in pruning of the components
            useACCoS (bool): Used to select whether or not ACCoS should be used 
        """
        self.core_tensor_G, self.components, optimal, ace_results = optimize_tensor_decomposition(self.X, self.Y, self.full_tensor_C, self.core_tensor_G, self.components, SNRs, ratios, useACCoS)
        return optimal, ace_results

    def deflate(self, deflate=(True, True), score_vector_matrix=False):
        r"""
        Deflate function. 
        
        The score vector is constructed. 
        The components are divided for X and Y.
        Finally, X and Y are deflated.

        Args:
            use_deflate ((bool, bool) tuple): Used to set whether (X, Y) should be deflated during each iteration
            score_vector_matrix (bool): Used to set whether the score vector should be created differently (as described in eBTTR paper, only tensor-matrix or tensor-vector)
        """
        idx_p = len(self.X.tensor.shape) - 1
        if score_vector_matrix:
            score_vector_t = tl.unfold(tl.tucker_to_tensor((self.X.tensor, [None] + self.components[:idx_p]), skip_factor=0, transpose_factors=True),0) @ np.linalg.pinv(fix_numpy_vector(tl.tensor_to_vec(self.core_tensor_G)).T) # type: ignore
            score_vector_t = fix_numpy_vector(score_vector_t / np.linalg.norm(score_vector_t))
        else:
            # compute svd, select left [0] matrix, select first singular vector 
            score_vector_t = fix_numpy_vector(scipy.linalg.svd(tl.unfold(tl.tucker_to_tensor((self.X.tensor, [None] + self.components[:idx_p]), skip_factor=0, transpose_factors=True),0), full_matrices=False)[0][:,0]) # type: ignore
        self.X.deflate(score_vector_t, self.components[:idx_p], deflate[0], self.core_tensor_G) # type: ignore
        self.Y.deflate(score_vector_t, self.components[idx_p:], deflate[1]) # type: ignore