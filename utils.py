"""
Utilities that do not require jax. Makes it easier to select the correct device for JAX
"""
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion
import numpy as np

def is_interactive() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    From https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def detect_local_minima_or_plateaus_1d(energy_values, energy_threshold=None):
    """
    Detect local minima in 1D energy landscape, including flat regions.
    Uses <= comparison to detect flat minima (plateaus).
    
    Args:
        energy_values: 1D array of energy values
        energy_threshold: Maximum energy value to consider (filters out infinite regions)
    
    Returns:
        minima_mask: Boolean array indicating minima positions
    """
    energy_values = np.array(energy_values)
    n = len(energy_values)
    
    # Filter out infinite/very high energy regions if threshold provided
    if energy_threshold is not None:
        valid_mask = energy_values <= energy_threshold
    else:
        valid_mask = np.isfinite(energy_values)
    
    # Initialize minima mask
    minima_mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if not valid_mask[i]:
            continue
            
        is_minimum = True
        current_energy = energy_values[i]
        
        # Check left neighbors
        for j in range(i-1, -1, -1):
            if not valid_mask[j]:
                break
            if energy_values[j] < current_energy:  # Found lower energy to the left
                is_minimum = False
                break
            elif energy_values[j] >= current_energy:  # Allow equal or higher, stop checking
                break
        
        if not is_minimum:
            continue
            
        # Check right neighbors
        for j in range(i+1, n):
            if not valid_mask[j]:
                break
            if energy_values[j] < current_energy:  # Found lower energy to the right
                is_minimum = False
                break
            elif energy_values[j] >= current_energy:  # Allow equal or higher, stop checking
                break
        
        minima_mask[i] = is_minimum
    
    return minima_mask