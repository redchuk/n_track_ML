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


def min_dist_TL(df, edges_stack):

    coord_x = np.asarray([i for i in range(edges_stack[0].shape[1])])
    coord_y = np.asarray([i for i in range(edges_stack[0].shape[0])])

    xarr = np.zeros((edges_stack[0].shape))
    xarr[:, :] = coord_x
    yarr = np.zeros((edges_stack[0].shape)).T
    yarr[:, :] = coord_y

    arr = np.sqrt((xarr - df['x']) ** 2 + (yarr.T - df['y']) ** 2)
    df['min_dist_pxs'] = arr[edges_stack[int(df['frame'])]].min()
    return df


def analyse_data(dots_df, nucl_masks, nucl_edges, pxsize, serum):

    dots_df['y_micron'] = dots_df['y'] * pxsize
    dots_df['x_micron'] = dots_df['x'] * pxsize

    areas = np.sum(nucl_masks, axis=(1, 2)) * pxsize ** 2
    perimeters = np.sum(nucl_edges, axis=(1, 2))
    dots_df['area_micron'] = areas[dots_df['frame']]
    dots_df['perimeter_au'] = perimeters[dots_df['frame']]
    dots_df['perimeter_au_norm'] = dots_df['perimeter_au'] * pxsize ** 2

    data_dist = dots_df.apply(min_dist_TL, axis=1, edges_stack=nucl_edges)
    data_dist['min_dist_micron'] = data_dist['min_dist_pxs'] * pxsize
    data_dist['script_version_git'] = get_git_hash()
    data_dist['serum'] = serum
    return data_dist


"""def aggregate():

    aggregated = pd.DataFrame(columns=["file",
                                       "script_version_git",
                                       "date",
                                       "guide",
                                       "time",
                                       "serum_conc_percent",
                                       "particle",
                                       "frame",
                                       "y", "x",
                                       "y_micron", "diff_y_micron",
                                       "x_micron", "diff_x_micron",
                                       "diff_xy_micron",
                                       "diff_xy_micron**2",
                                       "area_micron",
                                       "perimeter_au",
                                       "perimeter_au_norm",
                                       "min_dist_pxs",
                                       "min_dist_micron",
                                       "sqrt(area/pi)",
                                       "min_dist/(sqrt(area/pi))",
                                       "comment", "comment_long"])

    for i in glob.glob(o_path + "*"):
        filename = str(i).split('\\')[-1]
        data = pd.read_csv(i)
        data["date"] = filename[4:12]
        data["file"] = filename[4:-4]

        dots = data["particle"].unique()
        for i in dots:
            data_p = data[data["particle"] == i]
            data_p["diff_y_micron"] = data_p["y_micron"].diff()
            data_p["diff_x_micron"] = data_p["x_micron"].diff()
            data_p["diff_xy_micron**2"] = ((data_p["diff_x_micron"]) ** 2 + (data_p["diff_y_micron"]) ** 2)
            data_p["diff_xy_micron"] = data_p["diff_xy_micron**2"] ** .5
            data_p["sqrt(area/pi)"] = (data_p["area_micron"] / math.pi) ** .5
            data_p["min_dist/(sqrt(area/pi))"] = data_p["min_dist_micron"] / data_p["sqrt(area/pi)"]
            aggregated = aggregated.append(data_p, ignore_index=True, sort=False)
        print(filename + ' is ok')
    aggregated.set_index(["file", "particle"], inplace=True)

    aggregated.iloc[:, 0:26].to_csv(o_path[:-1] +
                                    "_aggregated/aggregated_n_track.csv")
"""

# todo: test below to be removed, or make bash script from it? main method?


inp_path = 'to_become_public/example_images/*ome.tif'
out_path = 'to_become_public/tracking_output/'


tstlst = glob.glob(inp_path, recursive=True)
print(tstlst)
filename = str(tstlst[1]).split('\\')[-1]
print(filename)

(tiff, meta) = ome.read_ometiff(tstlst[1])
xy_pxsize = float((meta.split(' ')[57]).split("\"")[1])
# todo: parse xml as Harri explained, https://www.geeksforgeeks.org/xml-parsing-python/
label_serum = int(filename.split('_')[1])

chrom = rescale_intensity(tiff[:,0,0,:,:])
lamin = rescale_intensity(tiff[:,0,1,:,:])
#plt.imshow(chrom[0,:,:])
#plt.imshow(lamin[0,:,:])

chrom_r, masks, edges = regis(chrom, lamin, fast = False)
data = find_dots_TL(chrom_r)
#tp.annotate(data.iloc[:4], chrom_r[0,:,:])

add_data = analyse_data(data, masks, edges, xy_pxsize, label_serum)

