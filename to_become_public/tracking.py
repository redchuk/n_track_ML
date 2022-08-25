import glob
import numpy as np
import math
import trackpy as tp
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from skimage import img_as_float
from skimage.filters import gaussian, threshold_multiotsu
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_erosion, remove_small_objects, remove_small_holes, disk
from apeer_ometiff_library import io as ome
from pystackreg import StackReg
import pandas as pd
import subprocess

inp_path = '**/*.tif'  # todo: fix path
out_path = '../tracking_output/'  # todo: fix path


def get_git_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def binary(dapi, gauss=1, classes=3, nbins=1000, min_size=15000, area_threshold=5000):
    dapi_g = img_as_float(gaussian(dapi, gauss))
    thresholds = threshold_multiotsu(dapi_g, classes=classes, nbins=nbins)
    mask_general = dapi_g > thresholds[0]
    clean_holes = remove_small_objects(mask_general, min_size=min_size)
    clean = remove_small_holes(clean_holes, area_threshold=area_threshold)
    return clean


def edge(binary):
    selem1 = disk(3)
    inside = binary_erosion(binary, selem1)
    binary[inside] = False
    return binary


def regis(chrom, lamin, fast=False):
    for t in range(chrom.shape[0]):
        ch = chrom[t]
        ch[np.invert(binary(lamin[t]))] = 0

    ch_mask = np.copy(chrom)
    for i in range(chrom.shape[0]):
        ch_mask[i] = (ch_mask[i] > 0)

    if fast == False:
        sr = StackReg(StackReg.RIGID_BODY)
        transformations = sr.register_stack(lamin, reference='previous')
        chrom = sr.transform_stack(chrom)
        ch_mask = sr.transform_stack(ch_mask)
        ch_mask = (ch_mask > 0.5)  # kludge
    else:
        pass

    ch_edges = np.copy(ch_mask)
    for k in range(ch_edges.shape[0]):
        ch_edges[k] = edge(ch_edges[k])

    return chrom, ch_mask, ch_edges


def find_dots_TL(stack, minmass=100000, diameter=11, search_range=20, memory=3, threshold=20):
    tp.quiet()
    f = tp.batch(stack, diameter=diameter, minmass=minmass)
    #todo: can it be just 'parameter' instead of 'parameter = parameter'?
    t = tp.link(f, search_range, memory=memory)
    t1 = tp.filter_stubs(t, threshold)
    return t1


# todo: test below to be removed, or make bash script from it?
tstlst = glob.glob(inp_path, recursive=True)
print(tstlst)
filename = str(tstlst[1]).split('\\')[-1]
print(filename)

(tiff, meta) = ome.read_ometiff(tstlst[1])
xy_pxsize = float((meta.split(' ')[57]).split("\"")[1])

chrom = rescale_intensity(tiff[:,0,0,:,:])
lamin = rescale_intensity(tiff[:,0,1,:,:])
#plt.imshow(chrom[0,:,:])
#plt.imshow(lamin[0,:,:])

chrom_r, masks, edges = regis(chrom, lamin, fast = False)
data = find_dots_TL(chrom_r)