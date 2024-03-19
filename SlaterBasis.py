import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.config import config;
config.update("jax_enable_x64", True)

@jit
def cartesian2Spherical(x,y,z):
    r = jnp.sqrt(x**2.0 + y**2.0 + z**2.0)
    theta = jnp.arccos(z/r)
    phi = jnp.arctan2(y,x)
    return r,theta,phi


def Dm(m):
    if m == 0:
        return (2*math.pi)**(-0.5)
    else:
        return math.pi**(-0.5)

def Clm(l,m):
    a = (2.0*l + 1.0)*math.factorial(l-m)
    b = 2.0*math.factorial(l+m)
    return (a/b)**0.5

@partial(jit, static_argnums=(0,1))
def Rn(n, alpha, x):
    if(n==1):
        return jnp.exp(-alpha*x)
    else:
        return jnp.power(x,n-1)*jnp.exp(-alpha*x)

def checkAssociatedLegendreRange(x):
    if ( x.any() < -1.0 or 1.0 < x.any()):
        raise Exception("The argument to associated legendre must be in [-1,1]")

@partial(jit, static_argnums=(0,1))
def associatedLegendre( n, m, x ):
    #print(type(x))

    # throw exception if x is not in [-1,1]
    jax.debug.callback(checkAssociatedLegendreRange,x)
    if ( m < 0 ):
        modM = abs(m)
        factor = ((-1.0)**m)*math.factorial(l-modM)/math.factorial(l+modM)
        return factor*associatedLegendre(n,modM,x)

    if ( n < m ):
        return jnp.zeros(len(x))

    #if ( x.any() < -1.0 or 1.0 < x.any()):
    #    raise Exception("The argument to associated legendre must be in [-1,1]")


    nx = len(x)
    cxM = jnp.ones(nx)
    somx2 = jnp.sqrt(1.0 - x*x)

    fact = 1.0
    for i in range ( 0, m ):
        cxM = - cxM * fact * somx2
        fact = fact + 2.0

    cx = cxM
    cxMplus = jnp.zeros(nx)
    if ( m != n ):
        cxMPlus1 = x * ( 2 * m + 1 ) * cxM
        cx = cxMPlus1

        cxPrev = cxMPlus1
        cxPrevPrev = cxM
        for i in range ( m + 2, n + 1 ):
            cx = ( ( 2 * i     - 1 ) * x * cxPrev \
                    + (   - i - m + 1 ) *     cxPrevPrev ) \
                    / (     i - m     )
            cxPrevPrev = cxPrev
            cxPrev = cx

    return ((-1.0)**m)*cx


@partial(jit,static_argnums=(0,))
def Qm(m, x):
    if m > 0:
        return jnp.cos(m*x)
    elif m == 0:
        return jnp.ones(x.shape)
    else:
        return jnp.sin(abs(m)*x)

def slaterFunctionHandle(n,l,m,alpha):
    def slaterBasisFunc(x):
        absm=abs(m)
        r,theta,phi = cartesian2Spherical(x[:,0],x[:,1],x[:,2])
        C = Clm(l,absm)*Dm(m)
        cosTheta = jnp.cos(theta)
        R = Rn(n, alpha, r)
        P = associatedLegendre(l, absm, cosTheta)
        Q = Qm(m, phi)
        return (C*R*P*Q)

    return slaterBasisFunc

#@partial(jit, static_argnums=(0,1,2,3))
#def slaterFunctionVal(n,l,m,alpha,x):
#    absm=abs(m)
#    r,theta,phi = cartesian2Spherical(x[:,0],x[:,1],x[:,2])
#    C = Clm(l,absm)*Dm(m)
#    cosTheta = jnp.cos(theta)
#    R = Rn(n, alpha, r)
#    P = associatedLegendre(l, absm, cosTheta)
#    Q = Qm(m, phi)
#    return (C*R*P*Q)



class SlaterPrimitive():
    def __init__(self, n, l, m, a):
        self.n = n
        self.l = l
        self.m = m
        self.a = a
        t1 = (2.0*a)**(n + 1.0/2.0)
        t2 = (np.math.factorial(2*n))**0.5
        self.nrm = t1/t2

    def alpha(self):
        return self.a

    def nlm(self):
        return self.n, self.l, self.m

    def normConst(self):
        return self.nrm

def getAtomSlaterBasis(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    lStringToIntMap = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4, 'h':5}
    basisList = []
    lchars = 'spdfgh'
    #ignore the first line
    for line in lines[1:]:
        words= line.split()
        if len(words) != 2:
            raise Exception("Invalid number of columns in file " + filename + ". Expects 2 values")

        nlString = words[0]
        alpha = float(words[1])
        n = 0
        l = 0
        lchar = nlString[-1].lower()
        found = lchar in lchars
        if found == False:
            raise Exception("Invalid l-character string found in file " + filename)

        n = int(nlString[:-1])
        l = lStringToIntMap[lchar]
        mList = []
        # NOTE: QChem even in the spherical form uses cartesian form for the s and p orbitals.
        # The ordering for cartesian orbitals are lexicographic - i.e., for p orbitals it's ordered
        # as x,y,z. While in the spherical form if one uses -l to +l ordering for the m quantum numbers for l=1 (p orbital),
        # then it translates to ordering the p orbitals as y,z,x.
        # To be consistent with QChem's ordering for p orbital, we do it in an
        # ad-hoc manner by ordering the m quantum numbers as 1,-1,0 for l=1 (p orbital).
        if l == 1:
            mList = [1, -1, 0]

        else:
            mList = list(range(-l,l+1))

        for m in mList:
            basis = SlaterPrimitive(n,l,m,alpha)
            basisList.append(basis)


    f.close()
    return basisList


def getSlaterFunctionVals(n, l, m, alpha, x, max_deriv = 0):
        returnVal = {}
        x_jnp = jnp.array(x)
        N = x_jnp.shape[0]
        slaterFunction = slaterFunctionHandle(n,l,m,alpha)
        #slaterFunction = partial(slaterFunctionVal, n,l,m,alpha)
        if max_deriv == 0:
            f = slaterFunction(x_jnp)
            returnVal[0] = np.array(f)
            return returnVal

        elif max_deriv == 1:
            f, df_fn = jax.vjp(slaterFunction,x_jnp)
            slaterFunction_sum = lambda x: jnp.sum(slaterFunction(x))
            df = df_fn(jnp.ones(f.shape))[0]
            returnVal[0] = np.array(f)
            returnVal[1] = np.array(df)
            return returnVal

        elif max_deriv == 2:
            f, df_fn = jax.vjp(slaterFunction,x_jnp)
            slaterFunction_sum = lambda x: jnp.sum(slaterFunction(x))
            df = df_fn(jnp.ones(f.shape))[0]
            hessian_fn = lambda x: jnp.sum(jax.grad(slaterFunction_sum)(x), axis=0)
            #hessian_fn = lambda x: jnp.sum(jax.grad(slaterFunction_sum)(x), axis=0)
            tmp, d2f_fn = jax.vjp(hessian_fn,x_jnp)
            dim = tmp.shape[0]
            I = jnp.eye(dim)
            d2f = jnp.zeros((N,3,3))
            d2f = d2f.at[:,0,:].set(d2f_fn(I[0,:])[0])
            d2f = d2f.at[:,1,:].set(d2f_fn(I[1,:])[0])
            d2f = d2f.at[:,2,:].set(d2f_fn(I[2,:])[0])
            returnVal[0] = np.array(f)
            returnVal[1] = np.array(df)
            returnVal[2] = np.array(d2f)
            return returnVal

        else:
            raise Exception("Slater function values and its derivatives only supported till second order.")

