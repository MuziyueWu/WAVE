import circle_functions as cf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io,filters, morphology, measure, draw,exposure
import pandas as pd

def ring_fitting(image, seg, features, width, threshold, show=False):
    """
    Fit each segmentation with a circle 
 
    Parameters
    ----------
    image : 2D array
        Raw tif image (single frame).
    seg : 2D array 
        Segmentations based on fluorescent intensities of the image. 
        Each integer corresponds to a unique segmentation and 0 is the background.
    features: pd.DataFrame
        Centroid,l abel, and frame of each segmentation.
    width: int
        Maximum radius of the circle.
    threshold: float([0,1])
        Fitting score threshold. Only rings with fitting score larger than threshold will be kept.
    show: boolean
        whether to plot the segmentated rings
    Returns
    -------
    result: DataFrame
        DataFrame with all the fitting results of qualified segmatations (radius of the fitted circle, intensity within the fitted circles,
        backgorund intensity, fitting score, area and perimeter of the fitted circle)
    """
    result = features.reset_index()
    result['r'], result['score'] = [np.nan, np.nan]
    local_seg_size = width * 2
    

    try:
        for r in range(0, len(result)):
            x_pos, y_pos = result.loc[r].x, result.loc[r].y
            if y_pos > width and x_pos > width and y_pos < image.shape[1]-width and x_pos < image.shape[0]-width: # skip corners
                y_min = int(y_pos - width)
                y_max = int(y_pos + width)
                x_min = int(x_pos - width)
                x_max = int(x_pos + width)
                local_data = image[x_min:x_max, y_min:y_max]
                local_seg = seg[x_min:x_max, y_min:y_max]
                local_seg_matched = local_seg == result.loc[r].label

                # Calculate fitting scores for all possible radiuses
                fitted_scores = [circle_opt(get_fitted_circle(radius, local_seg_size), local_data, local_seg_matched)[0] for radius in np.arange(1, width+1, 0.1)]
                
                # find the optimized circle and corresponding values
                opt_idx = np.argmax(fitted_scores)
                opt_r = 1 + 0.1 * opt_idx
                opt_circle = get_fitted_circle(opt_r, local_seg_size)
                opt_val, opt_intens, opt_bg = circle_opt(opt_circle, local_data, local_seg_matched)

                if opt_val > threshold and opt_r >= 2.5 and opt_r<6:
                    result.at[r,'r']= opt_r
                    result.at[r,'intensity']= opt_intens
                    result.at[r,'background']= opt_bg
                    result.at[r,'score'] = opt_val
                    result.at[r,'area'] = np.pi * np.power(opt_r, 2)
                    result.at[r,'perimeter'] = np.pi * opt_r *2
                    result.at[r,'diameter(nm)'] = opt_r*2*0.04*1000
                    result.at[r,'length(um)'] = np.pi*opt_r*2*0.04*1000
                    
                    if show:
                        plt.figure(figsize=(5, 5))
                        plt.subplot(131)
                        plt.imshow(local_data)
                        plt.subplot(132)
                        plt.imshow(local_seg_matched, vmin = 0, vmax = 1)
                        plt.subplot(133)
                        plt.imshow(opt_circle)
                        print(result.loc[r].label, opt_r, opt_val)
                        plt.show()
    except IndexError as e:
        raise e
        
    return result


def ring_fitting_wave_rac(image, rac,seg, features, width, threshold, show=False):
    """
    Fit WAVE segmentation with a circle 
 
    Parameters
    ----------
    image : 2D array
        Raw tif image from WAVE channel (single frame).
    rac : 2D array
        Raw tif image from rac channel (single frame).
    seg : 2D array 
        Segmentations based on fluorescent intensities of the image. 
        Each integer corresponds to a unique segmentation and 0 is the background.
    features: pd.DataFrame
        Centroid,l abel, and frame of each segmentation.
    width: int
        Maximum radius of the circle.
    threshold: float([0,1])
        Fitting score threshold. Only rings with fitting score larger than threshold will be kept.
    show: boolean
        whether to plot the segmentated rings
    Returns
    -------
    wave_line: 2D array
        local segmentation image containing one wave ring
    rac_line: 2D array
        corresponding local segmentation rac image
    """
    print('yes')
    result = features.reset_index()
    result['r'], result['score'] = [np.nan, np.nan]
    local_seg_size = width * 2
    wave_line = []
    rac_line = []
    try:
        for r in range(0, len(result)):
            x_pos, y_pos = result.loc[r].x, result.loc[r].y
            if y_pos > width and x_pos > width and y_pos < image.shape[1]-width and x_pos < image.shape[0]-width: # skip corners
                y_min = int(y_pos - width)
                y_max = int(y_pos + width)
                x_min = int(x_pos - width)
                x_max = int(x_pos + width)
                local_data = image[x_min:x_max, y_min:y_max]
                rac_data = rac[x_min:x_max, y_min:y_max]
                local_seg = seg[x_min:x_max, y_min:y_max]
                local_seg_matched = local_seg == result.loc[r].label

                # Calculate fitting scores for all possible radiuses
                fitted_scores = [circle_opt(get_fitted_circle(radius, local_seg_size), local_data, local_seg_matched)[0] for radius in np.arange(1, width+1, 0.1)]
                
                # find the optimized circle and corresponding values
                opt_idx = np.argmax(fitted_scores)
                opt_r = 1 + 0.1 * opt_idx
                opt_circle = get_fitted_circle(opt_r, local_seg_size)
                opt_val, opt_intens, opt_bg = circle_opt(opt_circle, local_data, local_seg_matched)

                if opt_val > threshold and opt_r >= 2.5 and opt_r<6:
                    if show:
                        plt.figure(figsize=(5, 5))
                        plt.subplot(141)
                        plt.imshow(local_data)
                        plt.subplot(142)
                        plt.imshow(local_seg_matched, vmin = 0, vmax = 1)
                        plt.subplot(143)
                        plt.imshow(opt_circle)
                        print(result.loc[r].label, opt_r, opt_val)
                        plt.show()
                    if opt_intens>0:
                        wave_line.append(local_data[:,int(width)])
                        rac_line.append(rac_data[:,int(width)])
    except IndexError as e:
        raise e
        
    return wave_line,rac_line

def seg_cells(image, threshold, sigma,min_size, max_size, plot = True):
    """
    Segment out regions with a given fluorescence threshold in a cell
 
    Parameters
    ----------
    image : 2D array
        Raw tif image (single frame).
    threshold: float
        threshold used for segmentation
    sigma: float
        Standard deviation for Gaussian kernel.The larger the value, the smoothier the gaussian fitting.
    min_size: float
        minimum size of the segmented region
    max_size: float
        maximum size of the segmented region
    plot: bool
        whether or not plot the segmentation image
    
    Returns
    -------
    region_properties: list of RegionProperties
        Each item describes one labeled region with the corresponding attributes. 
    label_mask_clean:2D array 
        Segmentations based on fluorescent intensities of the image. 
        Each integer corresponds to a unique segmentation and 0 is the background.
        
    """
    smooth = filters.gaussian(image,sigma,preserve_range=True)
    cell_mask = smooth > threshold
    label_mask = measure.label(cell_mask)
    region_properties = measure.regionprops(label_mask)  
    label_mask_clean = label_mask.copy()
    for r in region_properties:
        if r.area <= min_size or r.area >= max_size:
            label_mask_clean[label_mask_clean == r.label] = 0
    if plot:
        plt.figure(figsize = (10,10))
        plt.imshow(image)
        plt.contour(label_mask_clean,colors = 'r',linewidths = 0.7)

    region_properties = measure.regionprops(label_mask_clean,intensity_image=image)
    return region_properties,label_mask_clean

def circle_opt(fitted_circle, local_data, local_seg_matched, show=False):
    """
    Fit each segmentation with a circle 
 
    Parameters
    ----------
    fitted_circle: 2D array
        Circle to be fitted with the segmentation.
    local_data: 2D array
        Localized raw tif image corresponding to a single segmentation.
    local_seg_matched: 2D array
        Localized single segmentation.
    show: bool
        Debug flag for plotting.

    Returns
    -------
    fitting_score: float
        Based on area of the intersection and union of the fitting circle and local segmentation.
    intensity: float
        Total fluorescent intensities of the fitting area.
    background: float
        Background noise of the fitting area.
    
    """

    inter = np.bitwise_and(fitted_circle, local_seg_matched)
    union = np.bitwise_or(fitted_circle, local_seg_matched)
    excluded = union^1

    fitted_data = fitted_circle * local_data
    intensity = fitted_data.sum()
    background = (np.mean(excluded * local_data) * fitted_circle).sum()

    if show:
        plt.subplot(151)
        plt.imshow(local_data)
        plt.subplot(152)
        plt.imshow(local_seg_matched)
        plt.subplot(153)
        plt.imshow(fitted_circle)
        plt.subplot(154)
        plt.imshow(inter)
        plt.subplot(155)
        plt.imshow(union)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
        plt.show()
        print(inter.sum(), union.sum(), inter.sum() / union.sum())

    return inter.sum() / union.sum(), intensity, background
    
def get_fitted_circle(radius, size):
    """
    Generate a circle with given radius.
 
    Parameters
    ----------
    radius: float
        Radius of the fitted circle.
    size: int
        Size of the output array.
    
    Returns
    -------
    fitted_circle: 2D array
        Array of shape (size, size) where all 1s represent a circle.
    """

    center = size // 2
    x_local = np.linspace(0, size - 1, size)
    y_local = np.linspace(0, size - 1, size)
    x_local, y_local = np.meshgrid(x_local, y_local)
    distance_array = np.power(x_local - center, 2) + np.power(y_local - center, 2)

    return distance_array <= np.power(radius, 2)