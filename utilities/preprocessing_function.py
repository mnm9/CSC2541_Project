def preprocess_image(image, rgb, resize_shape = 256, channels_first = False):
    '''
    This function takes in a NumPy image and returns a preprocessed image
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