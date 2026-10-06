"""Model of the free-drift test, no lattice code: cold start (all links 1), zero force, one trajectory
of MD length t, so every link becomes exp(t P) and the links of a plaquette are independent. The
mean plaquette (1/3) Re Tr(U1 U2 U3^+ U4^+) is sampled with P drawn by each code's rule:
Chroma P = taproj(sqrt(1/2) G), G complex Gaussian (<|P|^2> = 4 per link); Grid with CPS_MD_TIME
P = i sqrt(2) sum_a c_a T_a, the same isotropic law with twice the variance (<|P|^2> = 8).
Chroma tau0 and Grid TRAJL = tau0 / sqrt(2) must give the same plaquette.

Usage: python3 free_drift_model.py [tau0 ...]   (default 0.1 0.354, the run_free_drift.sh values)
"""
import sys

import numpy as np

rng = np.random.default_rng(1)
M = 200000  # plaquette samples


def chroma_P(n):
    G = (rng.normal(size=(n, 3, 3)) + 1j * rng.normal(size=(n, 3, 3))) * np.sqrt(0.5)
    A = (G - np.conj(np.transpose(G, (0, 2, 1)))) / 2
    tr = np.trace(A, axis1=1, axis2=2) / 3
    return A - tr[:, None, None] * np.eye(3)


def grid_P(n):
    return chroma_P(n) * np.sqrt(2.0)


def expm_ah(A):
    w, v = np.linalg.eigh(-1j * A)  # A = i H, H hermitian
    return v @ (np.exp(1j * w)[:, :, None] * np.conj(np.transpose(v, (0, 2, 1))))


def plaq(Pgen, t):
    U = [expm_ah(t * Pgen(M)) for _ in range(4)]
    dag = lambda X: np.conj(np.transpose(X, (0, 2, 1)))
    W = U[0] @ U[1] @ dag(U[2]) @ dag(U[3])
    return np.real(np.trace(W, axis1=1, axis2=2)).mean() / 3


taus = [float(a) for a in sys.argv[1:]] or [0.1, 0.354]
print('mean |P|^2 per link: Grid %.3f  Chroma %.3f' % (
    np.mean(np.sum(np.abs(grid_P(M)) ** 2, axis=(1, 2))),
    np.mean(np.sum(np.abs(chroma_P(M)) ** 2, axis=(1, 2)))))
for tau in taus:
    tg = tau / np.sqrt(2)
    print(f'Chroma tau0={tau:<6}: plaq {plaq(chroma_P, tau):.4f}   Grid TRAJL={tg:.4f}: plaq {plaq(grid_P, tg):.4f}')
