### Common variables used throughout the simulation:


N: Total number of samples
n: number of variables

par: reference parameters vector

it: seed base of the noise generation



### Functions included in "BBRFcommon.py"



# 
X = Gen_noiseless(N,n,it,par)
 
  Generates the observation matrix X (N x (n+1))


# X = Add_noise(N,n,it,delta,X):
  Add noise to the "clean" observation matrix X based on the noise2signal ratio delta

# v = maxvar(S,n)
  Finds Maximum Admissible Noise variance from Samples Covariance matrix S 
  and store it in a vector v

# null(A, eps=1e-12)
  Extrancts null space from a matrix A

# simvol(A,n)
  Conputes the volume of a simplex defined by the vertices A (2D in this version of the code)

# A = OLS(S,n,norm=-1,val=-1)
  Computes the Ordinary Least Squares solutions compatible with the sample covariance
  matrix S and normalize to "val" the variable with index "norm" (-1 indicates the last one in Python)
  and store them in a matrix A

# x,y,L,U = BBRF(X,Sn,N,n,m)
  * Computes the Bounding Box Recursive Frisch Scheme from the noiseless observation matrix X,
  a noise covariance matrix Sn using recursion update of m samples.
  * Stores in x, y the coordinates of all the vertices of the simplex obtained at different iterations

# Data = BBRFmc_fix(N_mc,N,n,m,delta,par)
  * Performs a Monte Carlo simulation of N_mc runs and stores all the partial results in the list Data










