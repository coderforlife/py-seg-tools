"""
Image utility functions for memseg and related conversion scritps.
Utilizes numpy extensively. All images are numpy arrays in memory.
"""

from numpy import dtype, int8, uint8, int16, int32, int64, uint16, uint32, uint64, float32, float64

__all__ = [
    'IM_BYTE','IM_SBYTE','IM_SHORT','IM_SHORT_BE','IM_USHORT','IM_USHORT_BE','IM_INT','IM_INT_BE','IM_UINT','IM_UINT_BE','IM_LONG','IM_LONG_BE','IM_ULONG','IM_ULONG_BE',
    'IM_RGB24','IM_RGB24_STRUCT','IM_FLOAT','IM_DOUBLE',
    'is_rgb24', 'is_image_besides_rgb24', 'is_image',
    'gauss_blur', 'flip_up_down', 'bw', 'label', 'relabel', 'float_image', 'imread', 'imsave', 'imread_mat',
    ]

# The image types we know about
IM_BYTE      = dtype(uint8)
IM_SBYTE     = dtype(int8)
IM_SHORT     = dtype(int16).newbyteorder('<')
IM_SHORT_BE  = dtype(int16).newbyteorder('>')
IM_USHORT    = dtype(uint16).newbyteorder('<')
IM_USHORT_BE = dtype(uint16).newbyteorder('>')
IM_INT       = dtype(int32).newbyteorder('<')
IM_INT_BE    = dtype(int32).newbyteorder('>')
IM_UINT      = dtype(uint32).newbyteorder('<')
IM_UINT_BE   = dtype(uint32).newbyteorder('>')
IM_LONG      = dtype(int64).newbyteorder('<')
IM_LONG_BE   = dtype(int64).newbyteorder('>')
IM_ULONG     = dtype(uint64).newbyteorder('<')
IM_ULONG_BE  = dtype(uint64).newbyteorder('>')
IM_RGB24     = dtype((uint8,3))
IM_RGB24_STRUCT = dtype([('R',uint8),('G',uint8),('B',uint8)])
IM_FLOAT     = dtype(float32)
IM_DOUBLE    = dtype(float64)

def is_rgb24(im): return im.ndim == 2 and im.dtype == IM_RGB24_STRUCT or im.ndim == 3 and im.shape[2] == 3 and im.dtype == IM_BYTE
def is_image_besides_rgb24(im): return im.ndim == 2 and im.dtype in (
    IM_BYTE, IM_USHORT, IM_USHORT_BE, IM_UINT, IM_UINT_BE, IM_ULONG, IM_ULONG_BE,
    IM_SBYTE, IM_SHORT, IM_SHORT_BE, IM_INT, IM_INT_BE, IM_LONG, IM_LONG_BE, IM_FLOAT, IM_DOUBLE)
def is_image(im): return is_rgb24(im) or is_image_besides_rgb24(im)

def gauss_blur(im, sigma = 1.0):
    """
    Blur an image using a Gaussian blur. Requires SciPy.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(im, sigma = sigma)

def flip_up_down(im):
    """
    Flips an image from top-bottom. The returned value is a view, not a copy, so it will values changed in either will be reflected in the other.
    """
    from numpy import flipud
    return flipud(im)

def bw(im, threshold=1):
    """
    Convert image to black and white. The threshold is used to determine what is made black and white.
    If positive, every value at or above the threshold will be white and below it will be black.
    If negative, every value at or below the magnitude of the threshold will be white.
    If 0, the result will just be black.
    """
    return ((im>=threshold) if threshold>0 else (im<-threshold)).view(IM_BYTE)

def label(im):
    """
    Performs a connected-components analysis on the provided image. 0s are considered background.
    Any other values are connected into consecutively numbered contigous regions (where contigous is having a neighbor above, below, left, or right).
    """
    from scipy.ndimage import label
    return label(im)[0] # [1] is number of regions, obtainable later with im.max()

def relabel(im):
    """
    Creates a consecutively numbered image from an image that is already a set of labels.
    0 (or 0,0,0 for RGB) is the only value allowed to become 0 in the resulting image.
    Order is maintained. Note: Currently signed types where negative values are actually used are not supported.
    """
    # TODO: support using the same numbers across multiple slices
    from numpy import unique, insert, searchsorted
    # See scipy-lectures.github.io/advanced/image_processing/#measuring-objects-properties-ndimage-measurements for the unqiue/searchsorted method
    if is_rgb24(im):
        # RGB takes special care, and takes a little longer
        im = im.view(dtype=IM_RGB24_STRUCT).squeeze() # converting from IM_RGB24 to IM_RGB24_STRUCT has the 3rd dimension we need to get rid of
        values = unique(im)
        if tuple(values[0]) != (0, 0, 0): values = insert(values, 0, (0, 0, 0)) # make sure only 0 becomes 0 (may need to insert uint8(0) instead)
    elif is_image_besides_rgb24(im):
        # Make sure all the labels are consective starting from 1 and convert to IM_UINT
        values = unique(im)
        if values[0] < 0:
            raise ValueError('negative numbers')
            # TODO: negative values exist, a little harder (the code below 'works' but does not keep 0 in the 0 position...
            #pos0 = searchsorted(values, 0)
            #if pos0 == len(values) or values[pos0] != 0: values = insert(values, pos0, 0) # make sure only 0 becomes 0
        else:
            # only positive, easier
            if values[0] != 0: values = insert(values, 0, 0) # make sure only 0 becomes 0
            if im.dtype != IM_FLOAT and im.dtype != IM_DOUBLE and values[-1] == len(values) - 1: return im.astype(IM_UINT) # have consecutive numbers starting at 0 already, straight numeric conversion
    else: raise ValueError('im')
    n = len(values)
    if n < 256: dtype = IM_BYTE
    elif n < 65536: dtype = IM_USHORT
    elif n < 4294967296: dtype = IM_UINT
    else: raise OverflowError()
    return searchsorted(values, im).astype(dtype)

def float_image(im, in_scale = None, out_scale = (0.0, 1.0)):
    """
    Convert an image into a 32-bit floating-point image by scaling the data. Does not support RGB images.
    in_scale must be a length-2 list/tuple specifying lower and upper bounds.
    If in_scale is not provided or is None then the bounds of the underlying type is used (e.g. it is (0, 255) for IM_BYTE)
    out_scale is the range of floating-point numbers to map to, defaulting to 0.0 to 1.0
    """
    
    from numpy import empty, multiply, add
    
    # Process arguments
    if im.dtype == IM_RGB24_STRUCT: raise ValueError('im') # cannot float RGB image
    if in_scale == None:
        if   im.dtype == IM_BYTE:  in_scale = (0, 255)
        elif im.dtype == IM_SBYTE: in_scale = (-128, 127)
        elif im.dtype == IM_SHORT  or im.dtype == IM_SHORT_BE:  in_scale = (-32768, 32767)
        elif im.dtype == IM_USHORT or im.dtype == IM_USHORT_BE: in_scale = (0, 65535)
        elif im.dtype == IM_INT    or im.dtype == IM_INT_BE:    in_scale = (-2147483647, 2147483647)
        elif im.dtype == IM_UINT   or im.dtype == IM_UINT_BE:   in_scale = (0, 4294967295)
        elif im.dtype == IM_LONG   or im.dtype == IM_LONG_BE:   in_scale = (-9223372036854775807, 9223372036854775807)
        elif im.dtype == IM_ULONG  or im.dtype == IM_ULONG_BE:  in_scale = (0, 18446744073709551615)
        elif im.dtype == IM_FLOAT  or im.dtype == IM_DOUBLE:    in_scale = (im.min(), im.max())
        else: raise ValueError('im')
    elif len(in_scale) != 2 or in_scale[0] >= in_scale[1]: raise ValueError('in_scale')
    if len(out_scale) != 2 or out_scale[0] >= out_scale[1]: raise ValueError('out_scale')

    # Perform conversion
    in_min = in_scale[0]
    out_min = out_scale[0]
    k = float(out_scale[1] - out_min) / (in_scale[1] - in_min)
    out = empty(im.shape, dtype=IM_FLOAT)
    return add(multiply(im, k, out), out_min - in_min, out)

def imread_mat(filename, name = None):
    """
    Read an 'image' from a MATLAB .MAT file. The file can be any version. Files
    that are v7.3 require the h5py module. If no name is given, the first
    variable is taken.
    """
    try:
        # Try general first (doesn't work for v7.3+ files)
        # SciPy has this built in
        # Supports loading just the given variable name
        # Otherwise have to load all variables and skip special keys starting with "__" to find the variable to load
        # Loaded matracies are already arrays
        from scipy.io import loadmat
        if name == None:
            try:
                # Try to get first variable name without loading entire file (only supported in SciPy 0.12+)
                from scipy.io import whosmat
                keys in whosmat(file_name)
                if len(keys) == 0: raise KeyError()
                name = keys[0][0]
            except: pass
        x = loadmat(filename, variable_names = name)
        if name == None:
            name = '__' # we need to find first
            for name in x.iterkeys():
                if name[:2] != '__': break
            if name[:2] == '__': raise KeyError() # no variables
        return x[name] # can raise key error
    except NotImplementedError:
        # Try v7.3 file which is an HDF5 file
        # We have to use h5py for this (or PyTables...)
        # Always loads entire metadata (not just specific variable) but none of the data
        # Data needs to be actually loaded (.value) and transposed (.T)
        from h5py import File as HDF5File # TODO: if import error try using PyTables
        with HDF5File(filename, 'r') as x: # IOError if it doesn't exist or is the wrong format
            if name == None:
                try: name = x.iterkeys().next()
                except StopIteration: raise KeyError() # no variables
            return x[name].value.T # can raise key error

def imread(filename):
    """
    Read an image using SciPy (actually PIL) for any formats it supports.

    Additionally, the following extra formats are supported (extension must be right):
        MHA/MHD:    8-bit gray, 16-bit gray, 32-bit gray, 64-bit gray, float, double, 24-bit RGB
        MAT:        all formats, may not be image-like (requires h5py module for newer MAT files)
    Note: both only get the first "image" or data from the file.
    
    PIL Common Supported Formats: (not all-inclusive)
        PNG:  1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        TIFF: 1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB [not ZIP compressed]
        BMP:  1-bit BW, 8-bit gray, 24-bit RGB
        JPEG: 8-bit gray, 24-bit RGB
        IM:   all?
        
    See http://www.pythonware.com/library/pil/handbook/formats.htm for more details
    MHA/MHD code is implemented in the metafile module. MAT code is implemented in this module.
    """
    from os.path import splitext
    from scipy.misc import imread
    from metafile import imread_mha, imread_mhd

    ext = splitext(filename)[1].lower()
    if ext == '.mat':   return imread_mat(filename)
    elif ext == '.mha': return imread_mha(filename)[1]
    elif ext == '.mhd': return imread_mhd(filename)[1]
    else:               return imread(filename)
    
def imsave(filename, im):
    """
    Save an image. It will use SciPy (actually PIL) for any formats it supports.

    Additionally, the following extra formats are supported (extension must be right):
        MHA/MHD:    8-bit gray, 16-bit gray, 32-bit gray, 64-bit gray, float, double, 24-bit RGB
    
    PIL Common Supported Formats: (not all-inclusive)
        PNG:  1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        TIFF: 1-bit BW, 8-bit gray, 16-bit gray, 24-bit RGB
        BMP:  1-bit BW, 8-bit gray, 24-bit RGB
        JPEG: 8-bit gray, 24-bit RGB
        IM:   all?
    
    See thtp://www.pythonware.com/library/pil/handbook/formats.htm for more details
    MHA/MHD code is implemented in the metafile module.
    """
    from os.path import splitext
    from scipy.misc import imsave
    from metafile import imsave_mha, imsave_mhd
    
    ext = splitext(filename)[1].lower()
    if ext == '.mha':   imsave_mha(filename, im)
    elif ext == '.mhd': imsave_mhd(filename, im)
    else:               imsave(filename, im)
