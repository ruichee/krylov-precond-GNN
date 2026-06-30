import numpy as np


### CG takes an SPD matrix A (nxn) and solves Ax=b iteratively, finds exact x in max n iters
### The PCG solves the preconditioned system PAx = Pb, where the preconditioner P = M^-1
### CG is energy optimization, it steps in A-orthogonal search directions in consecutive k-th krylov subspace

### NOTE:
### Each CG step / iter is O(n^2) from matrix-vector multiplication
### Transposition is O(1) in numpy, no additional memory or copying is done, 
### the original matrix/vector is simply read from memory in a transposed order.


def preconditioned_conjugate_gradient(A: np.array[np.array], b: np.array, x0: np.array =None, P: np.array[np.array] =None,
                                      tol: float =1e-12, maxiter: int =10) -> np.array : 

    n = b.size          # dimension size of the problem
    if not P:           # no preconditioner specified, P = I (identity matrix)
        P = np.identity(n)
    if not x0:          # no initial condition specified, x0 = zero-vector
        x0 = np.zeros(n).T

    A = np.asarray(A, dtype=np.float64) 
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    r = b - A @ x       # initial residual vector
    d = P @ r           # initial search direction
    s = d.copy()        # initial search direction
    maxiter = min(maxiter, n)

    # CG requires that A is symmetric and positive definite (SPD)
    # check for symmetry here, assume positive-definite (consider using gershgorin circle theorem to check)
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric.")
    
    # PCG Iteration loop
    for k in range(maxiter):

        # obtain step size alpha in direction p, takes 1 step to get x_new and r_new
        # intermediate var q to reduce repeated O(n^2) matrix-vector multiplications
        q = A @ d
        alpha = (r @ s) / (d @ q)
        x_new = x + alpha * d
        r_new = r - alpha * q

        # check if tolerance criteria reached by residue to stop iteration
        if np.linalg.norm(r_new) / n < tol:
            break

        # obtain new (conjugate) search direction d_new
        # d_new is linear combo of all prevous d (spans k-th krylov subspace)
        s_new = P @ r_new
        beta = (r_new @ s_new) / (r @ s)
        d_new = s_new + beta * d

        # update variables for next iter
        r = r_new
        x = x_new
        d = d_new
        s = s_new

    return x_new



########## Test Cases for Demonstration ##########
'''
# Linear system variables
A = np.array([[42, 13, 0, 21], 
              [13, 54, 15, 0], 
              [0, 15, 54, 17],
              [21, 0, 17, 61]])
b = np.array([6, 8, 9, 2]).T
x0 = np.array([0, 0, 0, 0]).T
n_iter = 1

# identity preconditioner for benchmark
I = np.identity(4)
# choice of preconditioner for testing
P = np.linalg.inv(
    np.array([[42, 0, -12, 30],
              [0, 54, 0, -12],
              [-12, 0, 54, 0],
              [30, -12, 0, 61]]))

# exact solution for reference
x_exact = np.linalg.solve(A, b)
print(x_exact)

# run PCG algo with no precond
x1 = preconditioned_conjugate_gradient(A=A, b=b, x0=x0, P=I, tol=1e-5, maxiter=n_iter)
print(x1)

# run PCG algo with precond
x2 = preconditioned_conjugate_gradient(A=A, b=b, x0=x0, P=P, tol=1e-5, maxiter=n_iter)
print(x2)

# error, deviation of x from exact solution
print()
print(f"Initial L2-norm of x              : {np.linalg.norm(x0 - x_exact)}")
print(f"Final L2-norm of x (no precond)   : {np.linalg.norm(x1 - x_exact)}")
print(f"Final L2-norm of x (with precond) : {np.linalg.norm(x2 - x_exact)}")
print()

# original condition number of A
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Original Condition Number:          {np.max(eigenvalues) / np.min(eigenvalues)}")

# new condition number of PA
eigenvalues, eigenvectors = np.linalg.eig(P @ A)
print(f"Preconditioned Condition Number:    {np.max(eigenvalues) / np.min(eigenvalues)}")
print()
'''
