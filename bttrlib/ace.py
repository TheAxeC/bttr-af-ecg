import numpy as np
import tensorly as tl
from scipy.linalg import fractional_matrix_power

r"""
Functions for Automatic Parameter Estimation of BTTR
"""

class PrunedAllComponents(Exception):
    """
    Used when all components are pruned and thus nothing can be reconstructed in the decomposition
    """
    pass

def automatic_correlated_component_selection(X, Y, full_tensor_C, core_tensor_G, components_P):
    r"""
    Once the core tensor :math:`\underline{\mathbf{G}}^{(c)}` and factors :math:`\{\mathbf{P}^{(n)}\}^N_{n=2}` are extracted via ACE, we select only the relevant components in a fully automatic manner as well. The full process consists of two steps: scoring (Step 1) and grouping (Step 2). In Step 1, for each n-mode factor, the single rth component is scored using the R-squared test between :math:`\mathbf{x}_r^{(n)}` and :math:`\mathbf{y}` where

    .. math::
        \mathbf{x}_r^{(n)} = (\underline{\mathbf{X}} \times_2 \mathbf{P}^{(2)T} \times_3 ... \times_n \mathbf{P}_{\setminus r}^{(n)T} \times_{n+1} ... \times_N \mathbf{P}^{(N)T} )_{(1)} \text{vec}(\underline{\mathbf{G}}^{(c)}_{(n) \setminus r})

    with :math:`\underline{\mathbf{G}}^{(c)}_{(n) \setminus r} \in  \mathbb{R}^{(R_n - 1) \times (R_1 \times ... \times R_N)}` the n-mode matricization of the core tensor :math:`\underline{\mathbf{G}}^{(c)}` of which row r is removed. A lower score indicates a higher relevance of the associated (removed) component in the overall correlation between :math:`\underline{\mathbf{X}}` and :math:`\mathbf{y}`.

    In Step 2, we iteratively group the most relevant components. In order to select the smallest number of components that maximizes correlation, we start with the component with the lowest score in Step 1 and, then, iteratively add the one with the next lowest score. At each iteration, a new score is calculated using the R-squared test between :math:`\mathbf{x}_{\text{ind}}^{(n)}` and :math:`\mathbf{y}` where

    .. math::
        \mathbf{x}_{\text{ind}}^{(n)} = (\underline{\mathbf{X}} \times_2 \mathbf{P}^{(2)T} \times_3 ... \times_n \mathbf{P}_{\text{ind}}^{(n)T} \times_{n+1} ... \times_N \mathbf{P}^{(N)T} )_{(1)} \text{vec}(\underline{\mathbf{G}}^{(c)}_{(n) \text{ind}})

    with :math:`ind` the index of the first D components with lowest score and :math:`\mathbf{G}^{(c)}_{(n) \text{ind}} \in  \mathbb{R}^{ D \times (R_n \times ... \times R_N) }` the mode-n core tensor :math:`\underline{\mathbf{G}}^{(c)}` in which :math:`D` rows are selected. The selection process stops when the score starts to decrease. Finally, like ACE, given the new core tensor :math:`\underline{\mathbf{G}}^{(c)}` and factor matrices :math:`\{\mathbf{P}^{(n)}\}^N_{n=2}` with selected components, the score vector :math:`\mathbf{t}` is first computed as

    .. math::
        \mathbf{t} = (\underline{\mathbf{C}} \times_2 \mathbf{P}^{(2)T} \times_3 ... \times_N \mathbf{P}^{(n)T})_{(1)} \text{vec}(\underline{\mathbf{G}}^{(c)})

    and then normalized.

    Args:
        X (np.ndarray): Input tensor
        Y (np.ndarray): Secondary input tensor
        core_tensor_G (np.ndarray): Core tensor obtained from the tucker decomposition of full_tensor_C 
        components_P (np.ndarray): Factor matrices obtained from the tucker decomposition of full_tensor_C
    """
    N = X.shape

    return core_tensor_G, components_P

def update_G(full_tensor_C, components_P, snr):
    r"""
    At each iteration, the core tensor :math:`\underline{\mathbf{G}}` is updated using the soft-thresholding rule 
    as :math:`\underline{\mathbf{G}} = sgn(\underline{\mathbf{G}}) \times max\{abs(\underline{\mathbf{G}}) - \lambda, 0\}`

    See the following reference for a full overview of the algorithm:

    Yokota, Tatsuya, and Andrzej Cichocki. "Multilinear tensor rank estimation via sparse tucker decomposition." In 2014 Joint 7th International Conference on Soft Computing and Intelligent Systems (SCIS) and 15th International Symposium on Advanced Intelligent Systems (ISIS), pp. 478-483. IEEE, 2014.

    note: also the most computationally expensive function in BTTR

    Args:
        full_tensor_C (np.ndarray): The mode-1 cross-covariance tensor between X and Y
        components_P (np.ndarray): Factor matrices obtained from the tucker decomposition of full_tensor_C
        snr (float/int): parameter for use in the updating of the core tensor
    """
    epsilon = np.sum(full_tensor_C ** 2)*(10**(-snr/10))
    lambda_l = 0 # lower bound of lambda
    lambda_h = 1 # upper bound of lambda
    tolerance = 1e-8
    core_tensor_G = tl.tucker_to_tensor((full_tensor_C, components_P), transpose_factors=True)

    # Check the possibility of sparse solution
    err = np.sum((full_tensor_C - tl.tucker_to_tensor((core_tensor_G, components_P))) ** 2)
    if err > epsilon:
        return core_tensor_G
    
    # Normally repeat until convergence, but put limit on it
    for i in range(0, 100):
        core_tensor_G_h = tl.sign(core_tensor_G) * tl.clip((tl.abs(core_tensor_G) - lambda_h), 0)
        err = np.sum((full_tensor_C - tl.tucker_to_tensor((core_tensor_G_h, components_P))) ** 2)
        if err > epsilon:
            break
        else:
            lambda_l = lambda_h
            lambda_h = 2 * lambda_h
    
    # Optimal lambda should be between lambda_l and lambda_h
    for i in range(0, 1000):
        lambda_m = (lambda_l + lambda_h) / 2
        core_tensor_G_m = tl.sign(core_tensor_G) * tl.clip((tl.abs(core_tensor_G) - lambda_m), 0)
        err = np.sum((full_tensor_C - tl.tucker_to_tensor((core_tensor_G_m, components_P))) ** 2)
        if err > epsilon:
            lambda_h = lambda_m
        else:
            lambda_l = lambda_m
        
        if np.abs(lambda_h - lambda_l) < tolerance:
            break
    
    core_tensor_G_l = tl.sign(core_tensor_G) * tl.clip((tl.abs(core_tensor_G) - lambda_l), 0)
    return core_tensor_G_l

def pruning(core_tensor_G, components_P, ratio):
    r"""
    At each iteration, the threshold :math:`\tau \in [0, 100]` is used to reject unnecessary components from the n-mode :math:`S^{(n)} = \{r { | } 100 (1 - \frac{\sum_i{\mathbf{G}_{(n) (r, i)}}}{ \sum_{t,i}{\mathbf{G}_{(n) (t, i)}} }) \ge \tau\}`, :math:`\mathbf{P}^{(n)} = \mathbf{P}^{(n)} (:, S^{(n)})` and :math:`\mathbf{G}^{(n)} = \mathbf{G}^{(n)} (S^{(n)}, :)`.

    See the following reference for a full overview of the algorithm:

    Yokota, Tatsuya, and Andrzej Cichocki. "Multilinear tensor rank estimation via sparse tucker decomposition." In 2014 Joint 7th International Conference on Soft Computing and Intelligent Systems (SCIS) and 15th International Symposium on Advanced Intelligent Systems (ISIS), pp. 478-483. IEEE, 2014.

    Args:
        core_tensor_G (np.ndarray): Core tensor obtained from the tucker decomposition of full_tensor_C 
        components_P (np.ndarray): Factor matrices obtained from the tucker decomposition of full_tensor_C
        ratio (float/int): parameter for use in pruning of the components
    """
    N = len(components_P)
    RR = [components_P[n].shape[1] for n in range(0, N)]
    components_P_out = [None] * N
    for n in range(0, N):
        Gm = tl.unfold(core_tensor_G, n)
        gm = tl.sum(tl.abs(Gm), 1)
        ids = [k for k in range(0, Gm.shape[0]) if ((1 - gm[k] / tl.sum(gm)) * 100) > ratio]
        inv_ids = [k for k in range(0, Gm.shape[0]) if k not in ids]
        if len(inv_ids) == 0: raise PrunedAllComponents() # Added to resolve issue when all gets pruned
        RR[n] = len(inv_ids)
        Gm = Gm[inv_ids,:]
        components_P_out[n] = components_P[n][:,inv_ids]
        core_tensor_G = tl.fold(Gm, n, RR)
    return core_tensor_G, components_P_out

def update_P(full_tensor_C, core_tensor_G, components_P):
    r"""
    Is unnecessary due to components_P already being orthonormal
    """
    N = len(components_P)
    components_P_out = [None] * N
    for n in range(0, N):
        tmp_n = tl.unfold(tl.tucker_to_tensor((core_tensor_G, components_P), skip_factor=n), n)
        components_P_out[n] = tl.unfold(full_tensor_C, n) @ tmp_n.T
        components_P_out[n] = np.linalg.inv(fractional_matrix_power(components_P_out[n] @ components_P_out[n].T, 0.5)) @ components_P_out[n]  # type: ignore
    return components_P_out

def modified_pstd(full_tensor_C, core_tensor_G, components_P, snr, ratio):
    r"""
    .. pcode::
        :linenos:

        \begin{algorithm}
        \begin{algorithmic}
        \REQUIRE $\underline{\mathbf{X}} \in \mathbb{R}^{I_1 \times ... \times I_N}, \mathbf{y} \in \mathbb{R}^{I_1 \times 1}, \tau, $\textit{SNR}
        \ENSURE $\underline{\mathbf{G}} \in \mathbb{R}^{1 \times R_2 \times ... \times R_N}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$
        \\ \textit{Initialisation} :
        \STATE $\underline{\mathbf{C}} = \langle \underline{\mathbf{X}} , \mathbf{y} \rangle_{(1)} \in \mathbb{R}^{1 \times I_2 \times ... \times I_N}$
        \STATE \textbf{Initialisation of} $\{ \mathbf{P}^{(n)} \}^N_{n=2}$ and $\underline{\mathbf{G}}$ using HOOI on $\underline{\mathbf{C}}$
        \\ \textit{LOOP Process}
        \REPEAT
            \STATE \textbf{update} $\underline{\mathbf{G}}$ using \textit{SNR}
            \STATE \textbf{prune} $\{ \mathbf{P}^{(n)} \}^N_{n=2}$ and $\underline{\mathbf{G}}$ using $\tau$
        \UNTIL{convergence is reached}
        \RETURN $\underline{\mathbf{G}}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$ 
        \end{algorithmic}
        \end{algorithm}

    Args:
        full_tensor_C (np.ndarray): The mode-1 cross-covariance tensor between X and Y
        core_tensor_G (np.ndarray): Core tensor obtained from the tucker decomposition of full_tensor_C 
        components_P (np.ndarray): Factor matrices obtained from the tucker decomposition of full_tensor_C
        snr (float/int): parameter for use in the updating of the core tensor
        ratio (float/int): parameter for use in pruning of the components
    """
    maxiter = 100
    tol = 1e-16

    obj = tl.sum(tl.abs(core_tensor_G)) / tl.prod(full_tensor_C.shape)
    for itr in range(0, maxiter):
        """
            Pasting this code here just to document it
            The following commented code is present in the Matlab code [mySOTD_ARDL1.m line:34-46]
            It doesn't seem to have any impact on the output of the algorithm
            It is also not described within the papers
        """
        # for n in range(0, len(full_tensor_C.shape)):
        #     % Make the largest magnitude element be positive
        #     [~,loc] = max(abs(P{n}));
        #     for ii = 1:R(n)
        #         if P{n}(loc(ii),ii) < 0
        #             P{n}(:,ii) = P{n}(:,ii) * -1;
        # components_P = update_P(full_tensor_C, core_tensor_G, components_P)
        try:
            core_tensor_G = update_G(full_tensor_C, components_P, snr)
            core_tensor_G, components_P = pruning(core_tensor_G, components_P, ratio)
        except PrunedAllComponents:
            break

        obj2 = tl.sum(tl.abs(core_tensor_G)) / tl.prod(full_tensor_C.shape)
        if tl.abs(obj - obj2) < tol:
            break
        else:
            obj = obj2
    return core_tensor_G, components_P

def calculateBIC(full_tensor_C, core_tensor_G, components_P):
    r"""
    .. pcode::
        :linenos:
        
        \begin{algorithm}
        \begin{algorithmic}
        \REQUIRE $\underline{\mathbf{X}} \in \mathbb{R}^{I_1 \times ... \times I_N}, \mathbf{y} \in \mathbb{R}^{I_1 \times 1}$
        \ENSURE $\underline{\mathbf{G}}^{(X)} \in \mathbb{R}^{1 \times R_2 \times ... \times R_N}, \mathbf{t}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$

        \RETURN $BIC(\tau, \text{SNR} | \text{SNR}, \tau^*) = \log( \frac{\lVert\underline{\mathbf{C}} - \llbracket \underline{\mathbf{G}}^{(c)} ; \mathbf{P}^{(2)},...,\mathbf{P}^{(N)} \rrbracket\lVert_F}{s}) + \frac{\log(s)}{s} DF$
        \end{algorithmic} 
        \end{algorithm}

    Args:
        full_tensor_C (np.ndarray): The mode-1 cross-covariance tensor between X and Y
        core_tensor_G (np.ndarray): Core tensor obtained from the tucker decomposition of full_tensor_C 
        components_P (np.ndarray): Factor matrices obtained from the tucker decomposition of full_tensor_C
    """
    reconstructed = tl.tucker_to_tensor((core_tensor_G, components_P))
    df = np.count_nonzero(core_tensor_G)
    s = full_tensor_C.size
    xr = tl.sum((full_tensor_C - reconstructed)**2)
    return np.log(xr / s) + np.log(s)/s * df

def automatic_component_extraction(full_tensor_C, core_tensor_G, components_P, SNRs, ratios):
    r"""
    Automatic Component Extraction (ACE)

    .. pcode::
        :linenos:

        \begin{algorithm}
        \begin{algorithmic}
        \REQUIRE $\underline{\mathbf{X}} \in \mathbb{R}^{I_1 \times ... \times I_N}, \mathbf{y} \in \mathbb{R}^{I_1 \times 1}$
        \ENSURE $\underline{\mathbf{G}}^{(X)} \in \mathbb{R}^{1 \times R_2 \times ... \times R_N}, \mathbf{t}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$

        \STATE \textbf{Initialisation of} $\tau = 90,...,100$; \textit{SNR}$=1,...,50$

        \FOR {$\textit{SNR}_i$ in \textit{SNR}}
            \FOR {$\tau_j$ in $\tau$}
                \STATE $\underline{\mathbf{G}}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$ = \textit{mPSTD}($\underline{\mathbf{X}}$, $\mathbf{y}$, $\textit{SNR}_i$, $\tau_j$) 
                \STATE \textbf{calculate} BIC value corresponding to $\textit{SNR}_i$ and $\tau_j$ using calculateBIC
            \ENDFOR
            \STATE \textbf{select} $\tau^*$ = $\argmin_{\tau} \text{BIC}(\tau)$
            \STATE \textbf{calculate} BIC value corresponding to $\textit{SNR}_i$ and $\tau^*$ using calculateBIC
        \ENDFOR

        \STATE \textbf{select} $\textit{SNR}^*$ = $\argmin_{\textit{SNR}} \text{BIC}(\textit{SNR}, \tau^*)$
        \STATE $\underline{\mathbf{G}}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$ = \textit{mPSTD}($\underline{\mathbf{X}}$, $\mathbf{y}$, $\textit{SNR}^*$, $\tau^*$)

        \RETURN $\underline{\mathbf{G}}^{(X)}, \{ \mathbf{P}^{(n)} \}^N_{n=2}$ 
        \end{algorithmic} 
        \end{algorithm}

    Args:
        full_tensor_C (np.ndarray): The mode-1 cross-covariance tensor between X and Y
        core_tensor_G (np.ndarray): Core tensor obtained from the tucker decomposition of full_tensor_C 
        components_P (np.ndarray): Factor matrices obtained from the tucker decomposition of full_tensor_C
        SNRs (float/int list): list of parameters for use in the updating of the core tensor
        ratios (float/int list): list of parameters for use in pruning of the components
    """
    core_tensor_G_out, components_P_out = None, None
    optimal_bic = None
    ace_results = np.zeros((len(list(SNRs)), len(list(ratios))))
    optimal = ()
    i,j = 0,0
    for snr in SNRs:
        for ratio in ratios:
            tmp_core_tensor_G, tmp_components_P = modified_pstd(full_tensor_C, core_tensor_G, components_P, snr, ratio)
            bic = calculateBIC(full_tensor_C, tmp_core_tensor_G, tmp_components_P)
            if not optimal_bic or bic <= optimal_bic:
                optimal_bic = bic
                optimal = (snr, ratio, bic)
                core_tensor_G_out, components_P_out = tmp_core_tensor_G, tmp_components_P
            ace_results[i][j] = bic
            j += 1
        i += 1
        j = 0
    return core_tensor_G_out, components_P_out, optimal, ace_results

def optimize_tensor_decomposition(X, Y, full_tensor_C, core_tensor_G, components_P, SNRs=range(1,50), ratios=np.arange(95,99.9, 0.1), accos=False):
    r"""
    Base function to optimize the tucker decomposition

    Args:
        X (np.ndarray): Input tensor
        Y (np.ndarray): Secondary input tensor
        full_tensor_C (np.ndarray): The mode-1 cross-covariance tensor between X and Y
        core_tensor_G (np.ndarray): Core tensor obtained from the tucker decomposition of full_tensor_C 
        components_P (np.ndarray): Factor matrices obtained from the tucker decomposition of full_tensor_C
        SNRs (float/int list): list of parameters for use in the updating of the core tensor
        ratios (float/int list): list of parameters for use in pruning of the components
        accos (bool): Boolean to set whether ACCoS should be used
    """
    core_tensor, components, optimal, ace_results = automatic_component_extraction(full_tensor_C, core_tensor_G, components_P, SNRs, ratios)
    if accos: core_tensor_G, components_P = automatic_correlated_component_selection(X, Y, core_tensor_G, components_P)  # type: ignore
    return core_tensor, components, optimal, ace_results
    