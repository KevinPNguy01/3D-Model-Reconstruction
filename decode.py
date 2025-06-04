import numpy as np
import matplotlib.pyplot as plt

def decode_combined_code(folder, camera, code_threshold, object_threshold, display=False):
    """
    Computed a combined and masked gray code from a sequence of images.

    Parameters
    ----------
    folder : str
        Folder where the images are stored.

    camera : str
        Either 'C0' or 'C1' depending on the camera used.

    code_threshold : float
        Threshold to determine if a pixel's gray code is decodeable.

    object_threshold : float
        Threshold to determine if a pixel is part of the object mask.

    display : bool, optional
        If True, displays visualizations of the decoded gray code and masks.

    Returns
    -------
    masked_code : 2D numpy.array (dtype=float)
        Array containing the combined gray code values for each pixel, masked by the object.

    combined_mask : 2D numpy.array (dtype=bool)
        Array indicating which pixels were correctly decoded and are part of the object mask.
    """

    # Compute gray code and mask in each direction
    horizontal_code, horizontal_mask = decode(folder, camera, 0, code_threshold)
    vertical_code, vertical_mask = decode(folder, camera, 20, code_threshold)

    # Compute object mask
    object_mask = compute_object_mask(folder, camera, object_threshold)
    combined_mask = horizontal_mask & vertical_mask & object_mask

    # Combine horizontal and vertical codes and mask
    combined_code = 1024 * horizontal_code + vertical_code
    masked_code = np.zeros_like(combined_code, dtype=float)
    masked_code[combined_mask] = combined_code[combined_mask]

    if display:
        # display visualizaitons of the first scan
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        ax[0, 0].imshow(horizontal_code)
        ax[0, 0].set_title('Horizontal Code')
        ax[0, 1].imshow(vertical_code)
        ax[0, 1].set_title('Vertical Code')
        ax[0, 2].imshow(combined_code)
        ax[0, 2].set_title('Combined Code')
        ax[1, 0].imshow(horizontal_mask)
        ax[1, 0].set_title('Horizontal Mask')
        ax[1, 1].imshow(vertical_mask)
        ax[1, 1].set_title('Vertical Mask')
        ax[1, 2].imshow(combined_mask)
        ax[1, 2].set_title('Combined Mask')
        for axis in ax.flat:
            axis.axis('off')
        plt.figure(figsize=(6, 6))
        plt.imshow(masked_code)
        plt.title('Masked Combined Code')
        plt.axis('off')
        plt.show()

    return masked_code, combined_mask

def gray_to_binary(gray: int) -> int:
    """
    Convert a 10-bit gray code to binary.

    Parameters
    ----------
    gray : int
        The gray code value to convert.

    Returns
    -------
    int
        The corresponding binary value.
    """

    binary = gray
    mask = gray
    while mask != 0:
        mask >>= 1
        binary ^= mask
    return binary

def decode(folder, camera, start, threshold):
   """
   Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
   decode the binary sequence into a decimal value in (0,1023) for each pixel.
   Mark those pixels whose code is likely to be incorrect based on the user 
   provided threshold.  Images are assumed to be named "imageprefixN.png" where
   N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")

   Parameters
   ----------
   folder : str
      Folder where the images are stored

   camera : str
      Either C0 or C1 depending on the camera used
   
   start : int
      Starting index
      
   threshold : float
      Threshold to determine if a bit is decodeable
      
   Returns
   -------
   code : 2D numpy.array (dtype=float)
      Array the same size as input images with entries in (0..1023)
      
   mask : 2D numpy.array (dtype=logical)
      Array indicating which pixels were correctly decoded based on the threshold
   
   """
   
   # we will assume a 10 bit code
   nbits = 10
   
   # don't forget to convert images to grayscale / float after loading them in
   images = []
   for i in range(20):
      img = plt.imread(f"{folder}/frame_{camera}_{start+i:02d}_u.png")
      if img.ndim == 3:
         img = np.mean(img,axis=2)
      images.append(img)

   threshold_mask = np.ones_like(images[0],dtype=bool)
   code = np.zeros_like(images[0],dtype=float)

   # 10 binary images
   binary_images = np.zeros((nbits,images[0].shape[0],images[0].shape[1]),dtype=float)
   for i in range(nbits):
      # get the ith image
      binary_images[i] = images[i*2] - images[i*2+1]
      # update the mask
      threshold_mask[np.abs(binary_images[i]) < threshold] = False
      # threshold it
      binary_images[i][binary_images[i] < 0] = 0
      binary_images[i][binary_images[i] > 0] = 1
      
   # now we have 10 binary images, we need to convert them to a decimal value
   for i in range(nbits):
      code += binary_images[i] * (2**(nbits-i-1))
   code = np.vectorize(gray_to_binary)(code.astype(int))
        
   return code, threshold_mask

def compute_object_mask(folder, camera, threshold):
   """
   Compute the object mask from the decoded images.
   
   Parameters
   ----------
   folder : str
      Folder where the images are stored

   camera : str
      Either C0 or C1 depending on the camera used
   
   threshold : float
      Threshold to determine if a pixel is part of the object mask
      
   Returns
   -------
   mask : 2D numpy.array (dtype=logical)
      Array indicating which pixels are part of the object mask
   """
   
   bg_img = plt.imread(f"{folder}/color_{camera}_00_u.png")
   obj_img = plt.imread(f"{folder}/color_{camera}_01_u.png")
   subtracted = bg_img - obj_img
   obj_mask = np.ones_like(subtracted, dtype=bool)
   obj_mask[np.abs(subtracted) < threshold] = False
   obj_mask = np.any(obj_mask, axis=2)  # Convert to a 2D mask by checking all color channels

   return obj_mask