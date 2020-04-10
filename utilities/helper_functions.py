#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import pydicom
from skimage import morphology
from skimage import measure
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.cluster import KMeans
from statistics import mean
import re


# In[4]:


def generate_labels_feats(df, label_str, features_str):
    '''
    This function generates the feature and label data.
    Inputs:
        df:           DataFrame containing a columns with the Labels and the Features(filepath)
        label_str:    String referencing the column storing the labels in the dataframe
        features_str: String referencing the column storing the features in the dataframe
    Outputs:
        features and labels seperated out
    '''
    import pandas as pd
    
    return(df[features_str].apply(update_filepath), df[label_str])


# In[3]:


def update_filepath(text):
    '''
    This function updates the filepath.
    Inputs:
        text:          The original string
    Outputs:
        Replaced text
    '''
    old_text = '..'
    new_text = '/home/judy/devbox/cspine_hardware' 
    return(text.replace(old_text, new_text))


# In[6]:


def invert_pixel_array(output_loc, dicom_name, dicom_file, ref_image_1, hi_ro, lo_ro, hi_col, lo_col):
    '''
    This function ensures consistent photmetric interpretation across all input images. DICOM files
    with interpretation of MONOCHROME1 (ie white background) will be converted to MONOCHROME 
    (ie black background). This is achieved by adjusting pixel values if photometric interpretation is
    MONOCHROME1 with the following relationship:
                            pixel_inverted = pixel_range - pixel_original
    Inputs:
        output_loc:        String consisting of the Output Location where the converted images will be stored
        dicom_name:        String consisting of the DICOM file name
        dicom_file:        Raw DICOM file
        ref_image_1:       Reference Image
        hi_ro:             High threshold for rows (>0 - <=1)
        lo_ro:             Low threshold for rows (>0 - <=1)
        hi_col:            High threshold for cols (>0 - <=1)
        lo_col:            Low threshold for cols (>0 - <=1)
        
    Outputs:
        None - converted PNG files are stored in the appropriate location (as identified in provided 
        inputs)
    '''
    try:
        if dicom_file[0x28,0x04].value == 'MONOCHROME1': #If white background
            #Determine Pixel Range of Image:
            pixel_range = np.amin(dicom_file.pixel_array) + np.amax(dicom_file.pixel_array)
            
            #Find inverted pixel array: 
            inverted_array = pixel_range - dicom_file.pixel_array
            
            #Find maximum number of rows of inverted array
            max_rows = len(inverted_array)
            
            #Find the maximum number of colums of inverted array
            max_cols = len(inverted_array[0])
            
            #Save image as a PNG file, crop image to occur within row and col thresholds as specified
            #in inputs
            plt.imsave(output_loc + dicom_name + '.png', 
                       inverted_array[int(lo_ro*max_rows):int(hi_ro*max_rows), int(lo_col*max_cols):int(hi_col*max_cols)], 
                       cmap = plt.cm.gray)
        else: #If Black background
            max_rows = len(dicom_file.pixel_array)
            max_cols = len(dicom_file.pixel_array[0])
            plt.imsave(output_loc + dicom_name + '.png',
                       dicom_file.pixel_array[int(lo_ro*max_rows):int(hi_ro*max_rows), int(lo_col*max_cols):int(hi_col*max_cols)], 
                       cmap = plt.cm.gray)
    except: #On Error
        print('Error for: ' + dicom_name)


# In[9]:


def ipa_nosave(dicom_file, ref_image_1, hi_ro, lo_ro, hi_col, lo_col):
    '''
    This function ensures consistent photmetric interpretation across all input images. DICOM files
    with interpretation of MONOCHROME1 (ie white background) will be converted to MONOCHROME 
    (ie black background). This is achieved by adjusting pixel values if photometric interpretation is
    MONOCHROME1 with the following relationship:
                            pixel_inverted = pixel_range - pixel_original
    Inputs:
        dicom_file:        Raw DICOM file
        hi_ro:             High threshold for rows (>0 - <=1)
        lo_ro:             Low threshold for rows (>0 - <=1)
        hi_col:            High threshold for cols (>0 - <=1)
        lo_col:            Low threshold for cols (>0 - <=1)
        
    Outputs:
        None - converted PNG files are stored in the appropriate location (as identified in provided 
        inputs)
    '''
    try:
        if dicom_file[0x28,0x04].value == 'MONOCHROME1': #If white background
            #Determine Pixel Range of Image:
            pixel_range = np.amin(dicom_file.pixel_array) + np.amax(dicom_file.pixel_array)
            
            #Find inverted pixel array: 
            inverted_array = pixel_range - dicom_file.pixel_array
            
            #Find maximum number of rows of inverted array
            max_rows = inverted_array.shape[0]
            
            #Find the maximum number of colums of inverted array
            max_cols = inverted_array.shape[1]
            
        else: #If Black background
            max_rows = dicom_file.pixel_array.shape[0]
            max_cols = dicom_file.pixel_array.shape[1]

    except: #On Error
        print(f"Error inverting {dicom_file}")
    
    return dicom_file


def naive_contrast_stretching(dicom_file, low_thres, high_thres, clip_thres):
    '''
    This function adjusts window and level settings to retain artifacts that are relevant aka
    performs contrast stretching after performing Adaptive Histogram Equalization wherein several
    histograms to a distinct section of the images and uses them to resdistribute the lightness values
    of the image, helps improve local contrast and enhances edge definitions in each region of an 
    image.
    Inputs:
        dicom_file:        Raw DICOM file
        low_thres:         Low Threshold for percentile
        high_thres:        High Threshold for percentile
        clip_thres:        
    Outputs:
        Rescaled Image
    '''
    # Perform Adaptive Histogram Equalization
    img_file = exposure.equalize_adapthist(dicom_file.pixel_array, clip_limit = clip_thres)
    
    # Identify the low and high pixel values
    low_pixel, high_pixel = np.percentile(dicom_file.pixel_array, (low_thres, high_thres))
    
    # Rescale image intensities
    rescaled_img = exposure.rescale_intensity(dicom_file.pixel_array, in_range=(low_pixel, high_pixel))
    return(rescaled_img)


# In[11]:


def plot_img_hist(dicom_file):
    '''
    This function plots the image histogram ie the graphical representation of the tonal distribution
    in a digital image.
    Inputs:
        dicom_file:        Raw DICOM file
    Outputs:
        Image histogram
    '''
    plt.hist(dicom_file.pixel_array.flatten(), bins = 64)             #Calculate histogram of image
    plt.show()
    


# In[13]:


def create_folders(path_str):
    '''
    This function creates folders where output images (PNG verisons of DICOM files) will be stored.
    Inputs:
        path_str:         String consisting of the Path names
    Outputs:
        None - folders will be created at the path specified.
    '''
    os.mkdir(path_str)


# In[14]:


def Average(lst):
    '''
    This function finds the average given an input list    
    '''
    return (mean(lst))


# In[ ]:

def metrics_function(y_predicted, y_probs, y_true):
    '''
    This function takes an input of predictions and true values and returns weighted precision, recall, f1 scores,
    and AUC scores. 
    Inputs:
        y_predicted: NumPy array of shape (n_samples,) which contains predictions of categories
        y_probs: NumPy array of shape (n_samples, n_classes) which contains probabilities for each class
        y_true: NumPy array of shape (n_samples,) which contains actual labels for samples
    Outputs:
        f1_score: Weighted F1-score
        precision: Weighted Precision score
        recall: Weighted recall score
        auc: Weighted AUC score calculated using One-Versus-Rest Approach
        confusion_matrix: Confusion Matrix
    '''
    import sklearn.metrics
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    
    params = {
        'y_true': y_true,
        'y_pred': y_predicted,
        'average': 'weighted'
    }
    f1_score = sklearn.metrics.f1_score(**params)
    precision = sklearn.metrics.precision_score(**params)
    recall = sklearn.metrics.recall_score(**params)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true = y_true, y_pred = y_predicted)
    
    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(np.array(y_true).reshape(-1,1))
    auc = sklearn.metrics.roc_auc_score(y_true = y_encoded.toarray(), y_score = y_probs, average='weighted', multi_class = 'ovr')
    
    return f1_score, precision, recall, auc, confusion_matrix


# In[ ]:

def to_png(no_brands, dcom_data_path,sub_folder_list, crop_ind, process_ind, set_df_list,
            ref_image_1, LB_row, UB_row, LB_col, UB_col, clip_val, low_p, high_p):
    '''
    This function converts DICOM files to png and calls sub functions to crop or process (adaptive histogram equalization and intensity rescaling as specified). 
    Inputs:
        no_brands:          Number of Brands in dataset
        dcom_data_path:     Superior folder where DICOM files are stored: '/home/judy/devbox/cspine_hardware'
        sub_folder_list:    List of the subfolders where images are to be stored
        crop_ind:           Cropping indicator to specify if the image should be cropped (1) or not (0)
        process_ind:        Processing indicator to specify if the image should be processed (1) or not (0)
        set_df_list:        List of the training, validation and testing data sets
        ref_image_1:        A reference image #currently not used but could be used to determine pixel range
        LB_row:             Lower bound threshold (between 0 and 1) for pixel_array row
        UB_row:             Upper bound threshold (between 0 and 1) for pixel_array row
        LB_col:             Lower bound threshold (between 0 and 1) for pixel_array columns
        UB_col:             Upper bound threshold (between 0 and 1) for pixel_array columns
        clip_val:           Clipping Limit for Adaptive Histogram Equalization
        low_p:              Lower percentile limit for rescaling intensity
        high_p:             Upper percentile limit for rescaling intensity
    Outputs:
        DICOM images stored in specified  path after undergoing specified transformations (cropping and processing)
    '''     
    import re
    ind = 0
    for i in set_df_list:
        for j in range(no_brands):
            df = i[i['Label']==j+1]['filepath'].reset_index()
            count = 0
            for k in range(df.shape[0]):
                source_path = df['filepath'][k]
                source_path = dcom_data_path + source_path[2:]
                data = pydicom.dcmread(source_path)
                filename = (re.split('/', source_path)[-1]).split('.')[0]
                process_image_array(sub_folder_list[ind] + str(j+1) + '/', filename, data, crop_ind, process_ind, ref_image_1,
                                    LB_row, UB_row, LB_col, UB_col, clip_val, low_p, high_p)
        ind = ind + 1
    print('Finished!')


# In[ ]:

def process_image_array(output_loc, dicom_name, dicom_file, crop_ind, process_ind, ref_image_1, 
                       LB_row, UB_row, LB_col, UB_col, clip_val, low_p, high_p):
    '''
    This function processes the image array by ensuring consistent photmetric interpretation across all input
    images, cropping the image array if required between specified LB and UB bounds as well as performing
    adaptive histogram equalization and intentisty rescaling (ie   Naive contrast stretching). DICOM files
    with interpretation of MONOCHROME1 (ie white background) will be converted to MONOCHROME2 (ie black
    background). This is achieved by adjusting pixel values if photometric interpretation is
    MONOCHROME1 with the following relationship:
                            pixel_inverted = pixel_range - pixel_original
    Once done, if the user specified cropping, the image array is cropped between a specified upper and lower
    bound which are applied to the maximum rows and cols. Once cropped, if specified, adaptive histogram
    equalization with intensity rescaling is performed. Oncedone,images are stored as PNG files in the specified
    location.
    Inputs:
        output_loc:        String consisting of the Output Location where the converted images will be stored
        dicom_name:        String consisting of the DICOM file name
        dicom_file:        Raw DICOM file
        crop_ind:          Cropping indicator to specify if the image should be cropped (1) or not (0)
        process_ind:       Processing indicator to specify if the image should be processed (1) or not (0)
        ref_image_1:       Reference Image
        LB_row:             Lower bound threshold (between 0 and 1) for pixel_array row
        UB_row:             Upper bound threshold (between 0 and 1) for pixel_array row
        LB_col:             Lower bound threshold (between 0 and 1) for pixel_array columns
        UB_col:             Upper bound threshold (between 0 and 1) for pixel_array columns
        clip_val:           Clipping Limit for Adaptive Histogram Equalization
        low_p:              Lower percentile limit for rescaling intensity
        high_p:             Upper percentile limit for rescaling intensity
        
    Outputs:
        None - converted PNG files are stored in the appropriate location (as identified in provided 
        inputs)
    '''
    try:
        # Photometric Interpretation Inversion
        if dicom_file[0x28,0x04].value == 'MONOCHROME1': 
            pixel_range = np.amin(dicom_file.pixel_array) + np.amax(dicom_file.pixel_array) # Find Pixel Range
            output_array = pixel_range - dicom_file.pixel_array #Invert Image Array
        else:
            output_array = dicom_file.pixel_array #Store Original Image Array as Output Array
        
        # Find Max and Min values of array rows and columns:
        max_rows = len(output_array)
        max_cols = len(output_array[0])
                
        # Crop Image:
        if crop_ind == 1: #Crop image = Yes
            #Find LB and UB of rows and columns:
            LB_rows, UB_rows, LB_cols, UB_cols = compute_crop_bounds(max_rows, max_cols, LB_row, UB_row, LB_col,UB_col)
            # Modify Output Array:
            output_array = output_array[int(LB_rows):int(UB_rows), int(LB_cols):int(UB_cols)]
        
        #Process Image:
        if process_ind == 1: #Process image = Yes
            output_array = apply_naive_contrast_stretching(output_array, clip_val, low_p, high_p) # Apply adap hist equalization + rescale intensities
        
        # Save Image as PNG:
        plt.imsave(output_loc + dicom_name + '.png',output_array,cmap = plt.cm.gray)
        print('Finished')
    except: #log Files causing errors:
        print('Error for: ' + dicom_name)


# In[ ]:

def compute_crop_bounds(max_rows, max_cols, LB_row, UB_row, LB_col, UB_col):
    '''
    This function computes the cropping bounds of an input image given the maximum number of rows and columns
    of the image array, followed by the lower and upper bound thresholds for the rows and columns as specified
    within the input parameters.
    Inputs:
        max_rows:           The maximum number of rows of the Image Array
        max_cols:           The maximum number of columns of the Image Array
        LB_row:             Lower bound threshold (between 0 and 1) for pixel_array row
        UB_row:             Upper bound threshold (between 0 and 1) for pixel_array row
        LB_col:             Lower bound threshold (between 0 and 1) for pixel_array columns
        UB_col:             Upper bound threshold (between 0 and 1) for pixel_array columns
            
    Outputs:
        Lower bounds and upper bounds for rows and columns   
    '''
    
    LB_rows = LB_row * max_rows
    UB_rows = UB_row * max_rows
    LB_cols = LB_col * max_cols
    UB_cols = UB_col * max_cols
    return( LB_rows, UB_rows, LB_cols, UB_cols)

# In[ ]:

def apply_naive_contrast_stretching(image_array, clip_val, low_p, high_p):
    '''
    This function applies naive Contrast Stretching wherein the input image arrays undergo adaptive histogram
    equalization followed by rescaling of the intensities within a specified upper and lower bound.
    Inputs:
        image_array:        Pixel values of the image stored in an array
        clip_val:           Clipping Limit for Adaptive Histogram Equalization
        low_p:              Lower percentile limit for rescaling intensity
        high_p:             Upper percentile limit for rescaling intensity
            
    Outputs:
        Image array post contrast stretching   
    '''
    #Apply Adaptive Histogram Equilization
    new_image_array = exposure.equalize_adapthist(image_array, clip_limit = clip_val)
    #Determine low and high percentiles:
    p_lo, p_hi = np.percentile(new_image_array,(low_p, high_p))
    # Rescale Intensities:
    new_image_array = exposure.rescale_intensity(new_image_array, in_range=(p_lo, p_hi))
    return(new_image_array)


def prepare_image(image, rgb, resize_shape = 256, channels_first = False):
    '''
    This function takes in a NumPy image and returns in a friendly format for PIL conversion and/or tensor conversion
    Inputs:
        image: An X-ray image in NumPy format. The shape should be (Height, Width)
        rgb: Boolean indicating whether you want 3 channels (True) or not (False)
        resize_shape: Integer indicating the shape you want the image resized to
        channels_first: Boolean indicating whether you the output array shape to be 
                        (Channels, Height, Width) i.e. True or (Height, Width, Channels) 
                        i.e. False. Default is false
                        
    Outputs:
        A processed image based on the selections above. The output image is in NumPy 
        uint16 format. Should be fine if we need to convert to anything else
        The steps involved are:
            1. Resize the image to the shape specified. Default is 256x256
            2. Min-Max normalize the image to fall into the [0,255] range
            3. Return channels according to specification. If RGB, image is repeated 
               across all axes
            4. Image is returned with channels first or last, depending on specification
    '''
    import cv2
    import numpy as np

    img = image.copy()
    img = cv2.resize(img, (resize_shape, resize_shape)) # Resizing

    if img.any():
        min_, max_ = np.min(img), np.max(img) 
        img = ((img - min_)/(max_ - min_))*255 # Min max normalization
    
    # Creating a new channel based on 
    if channels_first:
        img = img[np.newaxis, ...]
    else:
        img = img[..., np.newaxis]
        
    if rgb:
        img = np.repeat(img, 3, 0 if channels_first else -1) # Returning a repeat along 3 axes if RGB
    
    return img.astype(np.uint16) # Return image otherwise


# Function to get mean and standard deviation for normalization
def get_training_mean_std_bmv(train_dataset):
    import numpy as np
    import torch
    
    means = []
    stds = []
    for i in range(len(train_dataset)):
        image = train_dataset.__getitem__(i)[0][0].numpy()
        mean, std = np.mean(image), np.std(image)
        means.append(mean)
        stds.append(std)

    return np.mean(means), np.mean(stds)