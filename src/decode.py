from src.visualize import visualize_graycode
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

def decode(folder, code_threshold, object_threshold, resultfile, display=False):
    """
    Decode the gray code from a sequence of images for both cameras and compute the object mask.

    Parameters
    ----------
    folder : str
        Folder where the images are stored.

    code_threshold : float
        Threshold to determine if a pixel's gray code is decodeable.

    object_threshold : float
        Threshold to determine if a pixel is part of the object mask.

    resultfile : str
        File name to save the decoded results.

    Returns
    -------
    C0 : 2D numpy.array (dtype=float)
        Combined gray code for camera C0.

    M0 : 2D numpy.array (dtype=bool)
        Mask indicating which pixels were correctly decoded and are part of the object mask for camera C0.

    C1 : 2D numpy.array (dtype=float)
        Combined gray code for camera C1.

    M1 : 2D numpy.array (dtype=bool)
        Mask indicating which pixels were correctly decoded and are part of the object mask for camera C1.

    """

    # If the result file already exists, load it
    if glob.glob(resultfile):
        print(f"Decoded gray code already exists at {resultfile}. Loading from file...")
        with open(resultfile, 'rb') as f:
            C0, M0, C1, M1 = pickle.load(f)
        return C0, M0, C1, M1
    
    # Otherwise, decode the gray code from the images
    print(f"Decoding gray code from images in {folder}...")
    C0, M0 = decode_combined_code(folder, "C0", code_threshold, object_threshold, display=display)
    C1, M1 = decode_combined_code(folder, "C1", code_threshold, object_threshold, display=display)

    # Save the results to the result file
    folder = os.path.dirname(resultfile)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(resultfile, 'wb') as f:
        pickle.dump((C0, M0, C1, M1), f)
    print(f"Decoded gray code saved to {resultfile}.")

    return C0, M0, C1, M1

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
    horizontal_code, horizontal_mask = decode_code(folder, camera, 0, code_threshold)
    vertical_code, vertical_mask = decode_code(folder, camera, 20, code_threshold)

    # Compute object mask
    object_mask = compute_object_mask(folder, camera, object_threshold)
    combined_mask = horizontal_mask & vertical_mask & object_mask

    # Combine horizontal and vertical codes and mask
    combined_code = 1024 * horizontal_code + vertical_code
    masked_code = np.zeros_like(combined_code, dtype=float)
    masked_code[combined_mask] = combined_code[combined_mask]

    if display:
        visualize_graycode(
            horizontal_mask, vertical_mask, object_mask, combined_mask,
            horizontal_code, vertical_code, combined_code, masked_code
        )

    return masked_code, combined_mask

def decode_code(folder, camera, start, threshold):
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
   
   # We will assume a 10 bit code
   nbits = 10
   
   # Load images and convert to grayscale if necessary
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
      # Get the ith image
      binary_images[i] = images[i*2] - images[i*2+1]
      # Update the mask
      threshold_mask[np.abs(binary_images[i]) < threshold] = False
      # Threshold it
      binary_images[i][binary_images[i] < 0] = 0
      binary_images[i][binary_images[i] > 0] = 1

   # Now we have 10 binary images, we need to convert them to a single "decimal" image
   for i in range(nbits):
      code += binary_images[i] * (2**(nbits-i-1))
   code = np.vectorize(gray_to_binary)(code.astype(int))
        
   return code, threshold_mask

def gray_to_binary(gray: int) -> int:
    """
    Convert a gray code to binary.

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