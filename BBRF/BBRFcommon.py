import numpy as np
import scipy
import time
import multiprocessing
#
#
from scipy import linalg, matrix
from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed

#
# DEFINE CLASSES
#
# Data structure for "fixed" Monte Carlo
class DataMCf(object):
	"Stores partial results of a simple Monte Carlo Simulation"
	def __init__(self, L, U, Stot, Atot, x, y):
		#super(DataMCf, self).__init__()
		self.L = L
		self.U = U
		self.Stot = Stot
		self.Atot = Atot
		self.x = x
		self.y = y	
#
#
# DEFINE FUNCTIONS
#
# Generate Noiseless Data
def Gen_noiseless(N,n,it,par):
    np.random.seed(it)
    U = np.random.randn(N,n) # Define Input
    y = U.dot(par) # Compute output
    X = np.hstack((U,y)) # Def OBSERVATION matrix
    return X
#
# Add noise to Noiseless Data
def Add_noise(N,n,it,delta,X):
    np.random.seed(it+N)
    Noise = np.random.randn(N,n+1)
    var = delta*np.diag(np.matmul(np.transpose(X),X)/N)
    X = X + Noise*var
    return X
 #
 # # Find Maximum Admissible Noise variance from Samples Covariance matrix
def maxvar(S,n):
    v = np.zeros((n+1,1))
    for i in range(0,n+1):
        Si = np.delete(S,i,0)
        Si = np.delete(Si,i,1)
        v[i] = np.linalg.det(S)/np.linalg.det(Si)
    return v
#
# Extract null space from matrix (based on SVD)
def null(A, eps=1e-12):
    u, s, vh = scipy.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)
#
# Compute Simplex Area (Volume)
def simvol(A,n):
	M = np.hstack((A,np.ones((n+1,1))))
	V=np.linalg.det(M)/2
	return V
# Performs ORDINARY LEAST SQUARES (OLS)
# returns a matrix whose columns are the OLS solutions 
def OLS(S,n,norm=-1,val=-1):
    #A = np.empty((1,n+1))
    A = np.zeros((n+1,1)).tolist()
    v = maxvar(S,n)
    for i in range(0,n+1):
        St = np.zeros((n+1,n+1))
        St[i,i] = v[i]
        Sh = S - St
        a = null(Sh)
        A = np.hstack((A,val*a/a[norm]))
    A = np.delete(A,0,1)
    A = np.delete(A,n,0)
    return A
#
#
def BBRF(X,Sn,N,n,m):
	b = int(N/m) # Total number of iterations
	#
    # Compute initial simplex
	X0 = X[0:m,:]
	S0 = np.matmul(np.transpose(X0),X0)/m + Sn
	A0 = np.transpose(OLS(S0,n))
	#
	# Initialize variables to store vertices of the simplices
	x = np.zeros((b+1,n+1))#[np.transpose(A0[:,0]).tolist()] 
	y = np.zeros((b+1,n+1))#[np.transpose(A0[:,1]).tolist()]
	x[0,:] = np.transpose(A0[:,0])#.append(np.transpose(A[:,0]).tolist())
	y[0,:] = np.transpose(A0[:,1])
	#
	# Initialize lower and upper bounds vectors
	L = np.zeros((b+1,n))#[0,0] 
	U = np.zeros((b+1,n))#[0,0]
	try:
		L[0,:] = A0.min(0)
		U[0,:] = A0.max(0)
	except ValueError:  #raised if `y` is empty.
		pass
	#
	# Perform Identification
	#for i in tqdm(range(1,b)):
	ti = time.time()
	for i in range(1,b+1):
		Xi = X[m*(i-1):m*i,:]
		Si = np.matmul(np.transpose(Xi),Xi)/m + Sn
		A = np.transpose(OLS(Si,n))
		x[i,:] = np.transpose(A[:,0])#.append(np.transpose(A[:,0]).tolist())
		y[i,:] = np.transpose(A[:,1])#.append(np.transpose(A[:,1]).tolist())
		if i==1:
			li = np.vstack((A.min(0),L))
			ui = np.vstack((A.max(0),U))
		li = np.vstack((A.min(0),L[i-1,:]))
		ui = np.vstack((A.max(0),U[i-1,:]))
		L[i,:] = li.max(0)
		U[i,:] = ui.min(0)
	#print('done.')
	t = time.time() - ti
	return x,y,L,U,t
#
#
#
def BBRFmc_fix(N_mc,N,n,m,delta,par):
	Data = []
	for i in tqdm(range(0,N_mc)):	
		X = Gen_noiseless(N,n,i,par)
		# Generate Noise
		#X = Add_noise(N,n,1,delta,X)
		var = delta*np.diag(np.matmul(np.transpose(X),X)/N)#[0,0,0]#
		Sn = np.diag(var)
		# Compute Total Simplex
		Stot = np.matmul(np.transpose(X),X)/N + Sn
		Atot = np.transpose(OLS(Stot,n))
		# Perform BOUNDING-BOX RECURSIVE FRISCH SCHEME estimation 
		x,y,L,U,t = BBRF(X,Sn,N,n,m)
		# Save partial results
		# Data.append(DataMCf(L,U,Stot,Atot,x,y))
		Data.append([L,U,Stot,Atot,x,y])
	return Data
#
#
def BBRFit(N,n,i,par,m,delta,Data):
	X = Gen_noiseless(N,n,i,par)
	# Generate Noise
	#X = Add_noise(N,n,1,delta,X)
	var = delta*np.diag(np.matmul(np.transpose(X),X)/N)#[0,0,0]#
	Sn = np.diag(var)
	# Compute Total Simplex
	Stot = np.matmul(np.transpose(X),X)/N + Sn
	Atot = np.transpose(OLS(Stot,n))
	# Perform BOUNDING-BOX RECURSIVE FRISCH SCHEME estimation 
	x,y,L,U,t = BBRF(X,Sn,N,n,m)
    # Compute Volume of the Solution sets
	V = (U[:,1]-L[:,1])*(U[:,0]-L[:,0])
	Vtot = simvol(Atot,n)
	Vit = np.zeros((int(N/m)))
	for i in range(0,int(N/m)):
		Ait = np.transpose([x[i],y[i]])
		Vit[i] = simvol(Ait,n)
	# Save partial results
	return L,U,Stot,Atot,x,y,V,Vit,Vtot
    
def BBRFmc_fix_par(N_mc,N,n,m,delta,par,Nc):
    Data = []
    inputs = range(N_mc)
    num_cores = multiprocessing.cpu_count()
    Data = Parallel(n_jobs=num_cores)(delayed(BBRFit)(N,n,i,par,m,delta,Data) for i in tqdm(inputs))
    return Data
#
#### Data analysis functions
#
# Process data from "fixed" Monte Carlo
#
def DPfix(Data,n,N_mc):
	# Process Lower and upper bounds
	Lavg = Data[0][0]
	Uavg = Data[0][1]
	Lmin = Data[0][0]
	Umin = Data[0][1]
	Lmax = Data[0][0]
	Umax = Data[0][1]
    # Process Solution set volumes
	Vavg = Data[0][6]
	Vmin = Data[0][6]
	Vmax = Data[0][6]
	Vitavg = Data[0][7]
	Vitmin = Data[0][7]
	Vitmax = Data[0][7]
	Vtotavg = Data[0][8]
	Vtotmin = Data[0][8]
	Vtotmax = Data[0][8]
	#
	print(Vitmin.shape,Vmin.shape)
	#
	for i in range(1,N_mc):
		Lavg = Lavg + Data[i][0]
		Uavg = Uavg + Data[i][1]
	    #
		Vavg = Vavg + Data[i][6]
		Vitavg = Vitavg + Data[i][7]
		Vtotavg = Vtotavg + Data[i][8]
	    #
		temp0 = np.vstack((Lmin[:,0],Data[i][0][:,0])).min(0)
		temp1 = np.vstack((Umin[:,0],Data[i][1][:,0])).min(0)
		temp2 = np.vstack((Lmax[:,0],Data[i][0][:,0])).max(0)
		temp3 = np.vstack((Umax[:,0],Data[i][1][:,0])).max(0)
        #
		Vmin = np.vstack((Vmin,Data[i][6])).min(0)
		Vmax = np.vstack((Vmax,Data[i][6])).max(0)
		Vitmin = np.vstack((Vitmin,Data[i][7])).min(0)
		Vitmax = np.vstack((Vitmax,Data[i][7])).max(0)
		Vtotmin = np.vstack((Vtotmin,Data[i][8])).min(0)
		Vtotmax = np.vstack((Vtotmax,Data[i][8])).max(0)
		#
		Lmin = np.vstack((temp0,np.vstack((Lmin[:,1],Data[i][0][:,1])).min(0)))
		Umin = np.vstack((temp1,np.vstack((Umin[:,1],Data[i][1][:,1])).min(0)))
		Lmax = np.vstack((temp2,np.vstack((Lmax[:,1],Data[i][0][:,1])).max(0)))
		Umax = np.vstack((temp3,np.vstack((Umax[:,1],Data[i][1][:,1])).max(0)))
		#
		Lmin = np.transpose(Lmin)
		Umin = np.transpose(Umin)
		Lmax = np.transpose(Lmax)
		Umax = np.transpose(Umax)    
		Vmin = np.transpose(Vmin)
		Vmax = np.transpose(Vmax)
		Vitmin = np.transpose(Vitmin)
		Vitmax = np.transpose(Vitmax)
		Vtotmin = np.transpose(Vtotmin)
		Vtotmax = np.transpose(Vtotmax)
        #
	Lavg = Lavg/N_mc
	Uavg = Uavg/N_mc
	#
	Vavg = Vavg/N_mc
	Vitavg = Vitavg/N_mc
	Vtotavg = Vtotavg/N_mc
	#
	return Lavg, Uavg, Lmin, Umin, Lmax, Umax, Vavg, Vmin, Vmax, Vitavg, Vitmin, Vitmax, Vtotavg, Vtotmin, Vtotmax
#### Drawing Functions
#
# Draw Solution box given low and up bounds
def drawbox(l,u):
	l = l[-1,:].tolist()
	u = u[-1,:].tolist()
	plt.fill([[l[0]],[l[0]],[u[0]],[u[0]]],[l[1],u[1],l[1],u[1]], 
		facecolor='k',alpha=0.5,edgecolor='k')
# Sparsify array in a logarithmic fashion
def logsparse(x):
	N = x.shape[0]
	o = findorder(N)
	xs = x[0:9]
	for i in range(1,o+2):
		idxi = pow(10,i)-1
		idxf = pow(10,i+1)-1
		if idxf>N:
			idxf = N + 1
		inter = pow(10,i-1)
		xs = np.append(xs,x[idxi:idxf:inter])
	return xs
#### General (Stupid) Functions
def findorder(x):
	i = 0
	while x>=10:
		x = x/10
		i +=1
	return i
#
def BBRFmod(X,Sn,N,n,m):
	b = int(N/m) # Total number of iterations
	#
    # Compute initial simplex
	X0 = X[0:m,:]
	S0 = np.matmul(np.transpose(X0),X0)/m + Sn
	A0 = np.transpose(OLS(S0,n))
	#
	# Initialize variables to store vertices of the simplices
	#
	# Initialize lower and upper bounds vectors
	L = np.zeros((b+1,n))#[0,0] 
	U = np.zeros((b+1,n))#[0,0]
	try:
		L[0,:] = A0.min(0)
		U[0,:] = A0.max(0)
	except ValueError:  #raised if `y` is empty.
		print('error: empty null space of the regressor')
		pass
	#
	# Perform Identification
	#for i in tqdm(range(1,b)):
	ti = time.time()
	for i in range(1,b+1):
		Xi = X[m*(i-1):m*i,:]
		Si = np.matmul(np.transpose(Xi),Xi)/m + Sn
		A = np.transpose(OLS(Si,n))
		if i==1:
			li = np.vstack((A.min(0),L))
			ui = np.vstack((A.max(0),U))
		li = np.vstack((A.min(0),L[i-1,:]))
		ui = np.vstack((A.max(0),U[i-1,:]))
		L[i,:] = li.max(0)
		U[i,:] = ui.min(0)
	#print('done.')
	t = time.time() - ti
	return L,U,t