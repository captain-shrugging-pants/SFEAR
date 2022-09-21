
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

from multiprocessing import Pool
from astropy.convolution import Gaussian2DKernel as astropyGaussian2D, convolve as astropyConv
from scipy import interpolate
from skimage.feature import blob_log, blob_doh, blob_dog
from skimage.color import rgb2gray
import copy
import random

nproc = multiprocessing.cpu_count()
qsfile = np.load('divergence_bfield_quiet_Sun.npz')
div = qsfile['div']
bq = qsfile['bfield']

def smooth(arr, sigma):
    return astropyConv(arr,astropyGaussian2D(sigma),preserve_nan=True,boundary='wrap')

with Pool(nproc) as p:
    bs = p.starmap(uf.smooth,zip(bq,[5]*len(bq)))
    divs = p.starmap(uf.smooth,zip(div,[7]*len(div)))
bs = np.asarray(bs) ; divs = np.asarray(divs)


def reject_cell(indices,bs_num):
    new_ind = []
    dist = 120 ; B_tol = 120
    hpix = int(dist/1.4/2)
    for idx,idd in enumerate(indices):
        id0,id1 = idd.astype(int)
        if abs(bs[bs_num,id0-hpix:id0+hpix,id1-hpix:id1+hpix]).max()<B_tol:
            new_ind.append([id0,id1])
    return np.array(new_ind)


def generate_inflow(blobs_log):
#     print(blobs_log)
    num = random.randrange(0,len(blobs_log))
    idx = blobs_log[num,:2].astype(int)
    idt = np.zeros((50,2))
    for ii in range(50):
        idt[ii,0] = int(np.random.normal(idx[0],5))
        idt[ii,1] = int(np.random.normal(idx[1]+10,14))
#         print(idt[ii])
    return idt


def gen_ind_list(divsm,samp_num):
    min_sigma=5 ; max_sigma=40 ; num_sigma=max_sigma-min_sigma+1
    image = -divsm
    image_gray = rgb2gray(image)
#     blobs_log = blob_log(image_gray, min_sigma=min_sigma, 
#                          max_sigma=max_sigma, num_sigma=num_sigma, 
#                          threshold=9e-7)
    blobs_log = blob_dog(image_gray, min_sigma=min_sigma, 
                         max_sigma=max_sigma, 
                         threshold=9.5e-6)
    id1 = blobs_log[:,0] ; id2 = blobs_log[:,1]
    mask = (id1>100)*(id1<400)*(id2>100)*(id2<400)
    blobs_log = blobs_log[mask]
    blobs_log = reject_cell(blobs_log[:,:2],samp_num)
    return blobs_log

with Pool(nproc) as p:
    gc.collect()
    indx_list = p.starmap(gen_ind_list,zip(divs,np.arange(len(divs))))
    
    

def gen_map(image,num):
    image = -image
    idt = generate_inflow(indx_list[num])
    idt1 = copy.copy(idt)
    xgrid = np.arange(512)
    for ii in range(len(idt)):
        idt1[ii,0] = len(xgrid)//2-idt[ii,0]
        idt1[ii,1] = len(xgrid)//2-idt[ii,1]
    tmp = 0
    for ii in range(len(idt1)):
        tmp += np.roll(np.roll(-image,int(idt1[ii,0]),axis=0),int(idt1[ii,1]),axis=1)
    return tmp

print('entering main loop')
r1 = 255-42 ; r2 = 255+43
maps_list = []
for ii in range(10):
    with Pool(nproc) as p:
        gc.collect()
        maps = p.starmap(gen_map, zip(divs,np.arange(len(divs))))
    maps = np.mean(maps,axis=0)
    maps_list.append(np.asarray(maps)[r1:r2,r1:r2])
maps_list = np.asarray(maps_list).astype(np.float32)
print(maps_list.shape)
print('saving file')
np.savez_compressed('model_flows/inflow.npz', arr=maps_list)
print('saved file')
