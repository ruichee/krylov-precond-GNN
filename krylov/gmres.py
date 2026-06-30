import numpy as np


### GMRES takes a general A (mxm) and solves Ax=b iteratively, findx exact x in max m iters
### the solution x is found by minimizing ||b-Ax|| in each consecutive n-th krylov subspace

### NOTE:
### need a way to store matrices as sparse matrices to reduce space complexity from O(m^2) to O(m)
### need to implement givens rotation to reduce the cost of least square solving step


def gmres(A: np.ndarray[np.ndarray], b: np.ndarray, x0: np.ndarray, tol: float =1e-6) -> np.ndarray:

    m = b.size                      # problem size / dimensions
    r = b - A @ x0                  # initial residual vector
    r_norm = np.linalg.norm(r)      # initial residual vector L2-norm

    Q = np.zeros((m, m+1))          # Q is mxm, additional (m+1)th column required 
    H = np.zeros((m+1, m))          # hessenberg matrix H is (m+1) x m
    q = r / r_norm                  # first unit vector of basis q1
    Q[:, 0] = q                     # column n is q_n -> basis vector for krylov subspace

    res_lst = [r_norm]              # list of residuals for tracking convergence behaviour

    for n in range(m):

        # carry out n-th step of Arnoldi iteration, which creates matrix Q_n
        # column space of Q_n forms an orthogonal basis of the n-th Krylov subspace
        q_n = Q[:, n]
        v = A @ q_n
        for j in range(n + 1):
            q_j = Q[:, j]
            h_jn = q_j @ v
            H[j, n] = h_jn
            v = v - h_jn * q_j

        h_new_n = np.linalg.norm(v)
        q_new = v / h_new_n
        H[n+1, n] = h_new_n
        Q[:, n+1] = q_new

        # solve least squares problem: find y to minimize || H_n@y - ||b||@e_1 ||
        # x is linear combo of basis vectors q, y is coefficient vector -> x = x0 + Q @ y
        # i.e. x is in the subspace spanned by Q, aka the orthogonalized Krylov subspace 
        e1 = np.array([[1] + [0]*m]).T
        y, res, _, _ = np.linalg.lstsq(H[:n+2, :n+1], r_norm*e1[:n+2])
        res_lst.append(np.sqrt(res[0]))

        # check if tolerance achieved for early termination, i.e. res < tol
        # obtain approximate solution x_n (best solution in n-th krylov subspace)
        # Note: res = ||r|| = ||b - Ax_n|| = || H_n@y - ||b||@e_1 ||, using the L2-norm
        if (res < tol): 
            print(f"Terminated at iteration {n+1} !")
            return x0 + Q[:, :n+1] @ y.flatten(), res_lst

    # obtain approximate solution x_m (solution in full m-th krylov subspace)
    # only reaches this case if full mth krylov subspace formed and tol not achieved
    # not expected to hit this case, since we expect number of iterations n << m
    print("Warning: tolerance value not achieved!")
    return x0 + Q[:, :-1] @ y.flatten(), res_lst



########## Test Cases for Demonstration ##########

# Linear system variables
n = 100
L = np.random.randn(n, n)
A = L @ L.T
b = np.random.randn(n)
x0 = np.array([0]*n)

# using our gmres function
x, res = gmres(A, b, x0)
'print(x)'

# exact value
x_exact = np.linalg.solve(A, b)
'print(x_exact)'

# scipy implementation of gmres
'''from scipy import sparse as spar
from scipy import linalg as la
x_scipy, info = spar.linalg.gmres(A, b, rtol=1e-5)
'print(x_scipy)'
print(info)'''

# comparing each gmres with exact
'''print(np.linalg.norm(x - x_exact)/n)
print(np.linalg.norm(x_scipy - x_exact)/n)'''

# plot convergence of our gmres algo
import matplotlib.pyplot as plt
plt.plot(res)
plt.semilogy()
plt.grid()
plt.xlim(0, n*1.1)
plt.ylim(0, res[0]*1.1)
plt.show()