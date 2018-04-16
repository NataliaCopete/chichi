import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import misc
import sys
from matplotlib import cm
from scipy import linalg


def transformada(arr):
  M=arr.shape[0]
  N=arr.shape[1]
  DFT=np.zeros((M,N),dtype=np.complex64)
  m=np.linspace(0,M-1,M, dtype=np.int16)
  n=np.linspace(0,N-1,N,dtype=np.int16)
  for k in range(M):
    
    expm=np.exp(-2.0*np.pi*1.0j*(m*float(k)/M))
    for l in range (N):
      
      expn=np.exp(-2.0*np.pi*1.0j*(n*float(l)/N))
      matrix=np.multiply(arr,expm[:, np.newaxis] *expn[np.newaxis,:])
     
      DFT[k,l]=matrix.sum()

  return (DFT)

def inversa(arr):
  M=arr.shape[0]
  N=arr.shape[1]
  DFT=np.zeros((M,N),dtype=np.complex64)
  m=np.linspace(0,M-1,M, dtype=np.int16)
  n=np.linspace(0,N-1,N,dtype=np.int16)
  for k in range(M):
    
    expm=np.exp(2.0*np.pi*1.0j*(m*float(k)/M))
    for l in range (N):
      
      expn=np.exp(2.0*np.pi*1.0j*(n*float(l)/N))
      matrix=np.multiply(arr,expm[:, np.newaxis] *expn[np.newaxis,:])
     
      DFT[k,l]=matrix.sum()

  return (DFT)



sigma= float(sys.argv[2])
img = misc.imread(sys.argv[1],mode='I')
dim1=img.shape[0]
dim2=img.shape[1]

filas=np.linspace(-dim1/10,dim1/10,dim1)
columnas=np.linspace(-dim1/10,dim1/10,dim2)
gauss1=np.exp(-filas**2.0/(2.0*(sigma)**2))
gauss2=np.exp(-columnas**2.0/(2.0*(sigma)**2))
gauss1=gauss1/np.trapz(gauss1)
gauss2=gauss2/np.trapz(gauss2)

kernel = gauss1[:, np.newaxis] * gauss2[np.newaxis,:]


dftimg2=transformada(img)
dftkernel2=transformada(kernel)
conv2=dftkernel2*dftimg2
inver2=inversa(conv2).real

def matriz(dimension):

  if(dimension%2.0!=0.0):
    medim1=(dimension-1)/2
  else:
    medim1=(dimension/2)-1

  h= np.zeros((dimension,dimension))

  for i in range(medim1+1,dimension):
    h[i,i-(medim1)]=1

  for i in range(medim1+1,dimension):
    h[i-(medim1),i]=1
  return h

I=matriz(dim1)
D=matriz(dim2)


inver3=I.dot(inver2.dot(D))

plt.imshow(inver3,cmap=cm.gist_gray)
plt.savefig('suave.png')




