import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import misc
from matplotlib import cm
from scipy import fftpack


img = misc.imread(sys.argv[1],mode='I')
corte= sys.argv[2]
dim1=img.shape[0]
dim2=img.shape[1]

def fr(M):  
  FREQ=np.zeros(M,dtype=float)
  
  for k in range(M):
    if (M%2==0.0):
      if(k<M/2):
        FREQ[k]=float(k)/M
      else:
        FREQ[k]= k/float(M)-1 
    else:
      if(k<=(M-1)/2):
        FREQ[k]=k/float(M)
      else:
        FREQ[k]=k/float(M)-1
  return FREQ
   

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


#cut=0.3536
#w=0.1768
cuth=0.5304
w=0.05
cutl=0.1768

if (corte=='alto'):
  filtro= np.zeros((dim1,dim2),dtype=np.float64)
  freq1=fr(dim1)
  freq2=fr(dim2)
  for i in range(dim1):
    for j in range(dim2):
      freq=np.sqrt((freq1[i])**2+(freq2[j])**2)
      if(freq > cuth+w):
        filtro[i,j]=1.0
      if(freq>=cuth-w and freq<=cuth+w):
        filtro[i,j]=0.5*(1-((np.sin(np.pi*(freq-cuth)))/(2*w)))

if (corte=='bajo'):
  filtro= np.zeros((dim1,dim2),dtype=float)
  freq1=fr(dim1)
  freq2=fr(dim2)
  for i in range(dim1):
    for j in range(dim2):
      freq=(freq1[i])**2+(freq2[j]**2)
      if(freq < cutl-w):
        filtro[i,j]=1.0
      if(freq>=cutl-w and freq<=cutl+w):
        filtro[i,j]=0.5*(1-((np.sin(np.pi*(freq-cutl)))/(2*w)))


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

filtronuevo=I.dot(filtro.dot(D))

##dftimg= fftpack.fft2(img,axes=(0,1))
dftimg= transformada(img)
dftimg=dftimg/(np.sum(dftimg))


conv=dftimg*filtronuevo
inver=inversa(conv).real

if (corte=='alto'):
 misc.imsave('altas.png',inver)

if(corte=='bajo'):
  misc.imsave('bajas.png',inver)



      

