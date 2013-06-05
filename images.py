"""
Image utility functions for memseg and related conversion scritps.
"""

from numpy import fromfile, dtype, int8, uint8, int16, uint16, uint32, float32

# The image types we know about
IM_BYTE      = dtype(uint8)
IM_SBYTE     = dtype(int8)
IM_SHORT     = dtype(int16).newbyteorder('<')
IM_SHORT_BE  = dtype(int16).newbyteorder('>')
IM_FLOAT     = dtype(float32)
IM_USHORT    = dtype(uint16).newbyteorder('<')
IM_USHORT_BE = dtype(uint16).newbyteorder('>')
IM_UINT      = dtype(uint32).newbyteorder('<')
IM_UINT_BE   = dtype(uint32).newbyteorder('>')
IM_RGB24     = dtype((uint8,3))
IM_RGB24_STRUCT = dtype([('R',uint8),('G',uint8),('B',uint8)])

class MRC:
    """
    Represents an MRC image. When loading only the header is loaded. 2D slices images are returned
    with the [] or when iterating. The number of slices is available with len(). The [] also accepts
    slice-notation and lists/tuples of indicies. Slices are only loaded as needed and are not
    cached. Thus requesting the same slice again will require an additional disk-read.

    The header properties are available as propties. They are all taken from MRC.FIELDS or
    MRC.FIELDS_OLD depending on the file version (most fields are shared between them). The labels
    are available as the 'labels' property. There is also a shape and dtype properties which are
    identical to the image array properties of returned slices.
    """
    BYTE    =  0 # 8 bit
    SHORT   =  1 # 16 bit, signed
    FLOAT   =  2 # 32 bit
    SHORT_2 =  3 # 32 bit, complex, signed
    FLOAT_2 =  4 # 64 bit, complex
    USHORT  =  6 # 16 bit, non-standard
    BYTE_3  = 16 # 24 bit, rgb, non-standard

    HEADER_LEN = 224
    LABEL_LEN = 80
    LABEL_COUNT = 10
    IMOD = 0x444F4D49
    SIGNED_BYTE_FLAG = 1
    MAP = 0x2050414D
    LITTLE_ENDIAN = 0x00004144
    BIG_ENDIAN = 0x00001717
    FIELDS_BASE = (
     'nx',     'ny',     'nz',      # number of columns, rows, and sections
     'mode',                        # pixel type (0-4, 6, 16)
     'nxstart','nystart','nzstart', # Starting point of sub-image (not used in IMOD)
     'mx',     'my',     'mz',      # grid size in X, Y, and Z
     'xlen',   'ylen',   'zlen',    # cell size, pixel spacing = xlen/mx, ...
     'alpha',  'beta',   'gamma',   # cell angles (not used in IMOD)
     'mapc',   'mapr',   'maps',    # map columns/rows/section in x/y/z (should always be 1,2,3)
     'amin',   'amax',   'amean',   # min/max/mean pixel value
     'ispf',                        # space group number (not used in IMOD)
     'next',                        # number of bytes in the extended header (called nsymbt in MRC standard)
     'creatid',                     # used to be an ID, now always 0
     'nint', 'nreal',               # meaning is dependent on extended header format
     'imodStamp', 'imodFlags',      # if imodStamp == 0x444F4D49 (IMOD) and imodFlags == 1 then bytes are signed
     'idtype', 'lens', 'nd1', 'nd2', 'vd1', 'vd2', # Imaging attributes
     'tiltangles0', 'tiltangles1', 'tiltangles2', 'tiltangles3', 'tiltangles4', 'tiltangles5', # Imaging axis
     )
    FIELDS = FIELDS_BASE + (
     'xorg',   'yorg',   'zorg',    # origin of image
     'cmap', 'stamp',               # for detecting file type, cmap == 0x2050414D (MAP ) and stamp == 0x00004441 or 0x00001717 for little/big endian
     'rms',                         # the RMS deviation of densities from mean density
     'nlabl',                       # number of meaningful labels
     )
    FIELDS_OLD = FIELDS_BASE + (
     'nwave', 'wave1', 'wave2', 'wave3', 'wave4', 'wave5', # Wavelengths
     'xorg',   'yorg',   'zorg',    # origin of image
     'nlabl',                       # number of meaningful labels
     )

    def __init__(self, filename):
        from struct import unpack

        self.file = open(filename, "rb")
        raw = self.file.read(MRC.HEADER_LEN)

        # Parse Header
        vers = unpack('<ii', raw[208:216])
        endian = '<'
        if vers[0] == MRC.MAP:
            if vers[1] == MRC.BIG_ENDIAN:
                endian = '>'
            elif vers[1] != MRC.LITTLE_ENDIAN:
                raise IOError('MRC file is invalid (stamp is 0x%08x)' % self.header['stamp'])
            h = dict(zip(MRC.FIELDS, unpack(endian + '10i6f3i3fiih30xhh20xii6h6f3f2ifi', raw)))
        else:
            h = dict(zip(MRC.FIELDS_OLD, unpack('<10i6f3i3fiih30xhh20xii6h6f6h3fi', raw)))
        self.header = h

        if h['nx'] < 0 or h['ny'] < 0 or h['nz'] < 0:          raise IOError('MRC file is invalid (dims are %dx%dx%d)' % (h['nx'], h['ny'], h['nz']))
        if h['mapc'] != 1 or h['mapr'] != 2 or h['maps'] != 3: raise IOError('MRC file is has an unusual ordering (%d, %d, %d)' % (h['mapc'], h['mapr'], h['maps']))
        if h['next'] < 0:                                      raise IOError('MRC file is invalid (extended header size is %d)' % h['next'])
        if h['nlabl'] < 0 or h['nlabl'] > 10:                  raise IOError('MRC file is invalid (the number of labels is %d)' % h['nlabl'])

        # TODO: validate these:
        ##'mx',   'my',   'mz',   # grid size in X, Y, and Z
        ##'xlen', 'ylen', 'zlen', # cell size, pixel spacing = xlen/mx, ...
        ##'xorg', 'yorg', 'zorg', # origin of image

        # Read Labels
        h['labels'] = [self.file.read(MRC.LABEL_LEN) for _ in range(0, h['nlabl'], 1)]

        # Deterimine data type
        if   h['mode'] == MRC.BYTE:   self.dtype = IM_SBYTE if h['imodStamp'] == MRC.IMOD and h['imodFlags'] == MRC.SIGNED_BYTE_FLAG else IM_BYTE
        elif h['mode'] == MRC.SHORT:  self.dtype = IM_SHORT.newbyteorder(endian)
        elif h['mode'] == MRC.FLOAT:  self.dtype = IM_FLOAT
        elif h['mode'] == MRC.USHORT: self.dtype = IM_USHORT.newbyteorder(endian)
        elif h['mode'] == MRC.BYTE_3: self.dtype = IM_RGB24
        elif h['mode'] == MRC.SHORT_2 or h['mode'] == MRC.FLOAT_2:
            raise IOError('MRC file uses a complex format which is not supported')
        else:
            raise IOError('MRC file is invalid (mode is %d)' % h['mode'])

        # Precompute these for getting sections fast
        self.shape = (h['ny'], h['nx']) + self.dtype.shape
        self.data_offset = MRC.HEADER_LEN + MRC.LABEL_LEN * MRC.LABEL_COUNT + h['next']
        self.section_size = h['ny'] * h['nx']
        self.section_data_size = self.section_size * self.dtype.itemsize

    def __del__(self): self.close()
    def close(self):
        if self.file:
            self.file.close()
            del self.file
            self.file = None

    def __getattr__(self, name):
        if name == 'header' or not self.header.has_key(name): raise AttributeError(name)
        return self.header[name]

    def __get_section(self, i):
        self.file.seek(self.data_offset + i * self.section_data_size)
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.shape)
    def __get_next_section(self):
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.shape)

    def __len__(self): return self.header['nz']
    def __getitem__(self, index):
        nz = self.header['nz']
        if isinstance(index, (int, long)):
            if index < 0: index = nz - index
            if index >= nz: raise ValueError('index')
            return self.__get_section(index)
        elif isinstance(index, slice):
            # TODO: should accept lists in addition to slices
            # TODO: slices should return an iterable generator thing instead of images preloaded
            if index.start == None: index.start = 0
            if index.stop == None or index.stop == 2147483647: index.stop = nz
            if index.step == 0 and index.start < index.stop and index.start < 0 or index.start >= nz: raise ValueError('index')
            if index.step == 1 or index.step == None:
                if index.stop > nz: raise ValueError('index')
                if index.start >= index.stop: return []
                l = [self.__get_section(index.start)]
                for _ in range(index.start + 1, index.stop): l.append(self.__get_next_section())
                return l
            else:
                r = range(index.start, index.stop, index.step)
                if len(r) == 0: return []
                if index.step > 0 and r[-1] >= nz or index.step < 0 and r[-1] < 0: raise ValueError('index')
                l = []
                for z in r: l.append(self.__get_section(z))
                return l
        else: raise TypeError('index')
    def __iter__(self):
        yield self.__get_section(0)
        for _ in range(1, self.header['nz']): yield self.__get_next_section()


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

def create_labels(im):
    """
    Creates a consecutively numbered IM_UINT image from an image.
    0 (or 0,0,0 for RGB) is the only value allowed to become 0 in the resulting image.
    Order is maintained. Note: Currently signed types are not supported.
    """
    # TODO: support using the same numbers across multiple slices
    # TODO: support running connected components code on BW data ( scipy.ndimage.label() )
    from numpy import unique, insert, searchsorted
    # See scipy-lectures.github.io/advanced/image_processing/#measuring-objects-properties-ndimage-measurements for the unqiue/searchsorted method
    if im.ndim == 2 and im.dtype == IM_RGB24_STRUCT or im.ndim == 3 and im.shape[2] == 3 and im.dtype == IM_BYTE: # IM_RGB24 can't be directly detected
        # RGB takes special care, and takes a little longer
        im = im.view(dtype=IM_RGB24_STRUCT).squeeze() # converting from IM_RGB24 to IM_RGB24_STRUCT has the 3rd dimension we need to get rid of
        values = unique(im)
        if tuple(values[0]) != (0, 0, 0): values = insert(values, 0, uint8(0)) # make sure only 0 becomes 0
        # can't do the cheat where we would jump out now...
    elif im.ndim == 2 and im.dtype in (IM_BYTE, IM_USHORT, IM_USHORT_BE, IM_UINT, IM_UINT_BE): # IM_SBYTE, IM_SHORT, IM_SHORT_BE, IM_FLOAT?
        # Make sure all the labels are consective starting from 1 and convert to IM_UINT
        values = unique(im)
        #if values[0] < 0:
        #    # TODO: negative values exist, a little harder (the code below 'works' but does not keep 0 in the 0 position...
        #    pos0 = searchsorted(values, 0)
        #    if pos0 == len(values) or values[pos0] != 0: values = insert(values, pos0, 0) # make sure only 0 becomes 0
        #    # can't do the cheat where we would jump out now...
        #else:
            # only positive, easier
        if values[0] != 0: values = insert(values, 0, 0) # make sure only 0 becomes 0
        if im.dtype != IM_FLOAT and values[-1] == len(values) - 1: return im.astype(IM_UINT) # have consecutive numbers starting at 0 already, straight numeric conversion
    else: raise ValueError('im')
    return searchsorted(values, im).astype(IM_UINT)

def float_image(im, in_scale = None, out_scale = (0.0, 1.0)):
    """
    Convert an image into a 32-bit floating-point image by scaling the data. Does not support RGB images.
    in_scale must be a length-2 list/tuple specifying lower and upper bounds.
    If in_scale is not provided or is None then the bounds of the underlying type is used (e.g. it is (0, 255) for IM_BYTE)
    out_scale is the range of floating-point numbers to map to, defaulting to 0.0 to 1.0
    """
    
    from numpy import vectorize
    
    # Process arguments
    if im.dtype == IM_RGB24_STRUCT: raise ValueError('im') # cannot float RGB image
    if in_scale == None:
        if   im.dtype == IM_BYTE:  in_scale = (0, 255)
        elif im.dtype == IM_SBYTE: in_scale = (-128, 127)
        elif im.dtype == IM_USHORT or im.dtype == IM_USHORT_BE: in_scale = (0, 65535)
        elif im.dtype == IM_SHORT  or im.dtype == IM_SHORT_BE:  in_scale = (-32768, 32767)
        elif im.dtype == IM_UINT   or im.dtype == IM_UINT_BE:   in_scale = (0, 4294967295)
        elif im.dtype == IM_FLOAT: in_scale = (im.min(), im.max())
        else: raise ValueError('im')
    elif len(in_scale) != 2 or in_scale[0] >= in_scale[1]: raise ValueError('in_scale')
    if len(out_scale) != 2 or out_scale[0] >= out_scale[1]: raise ValueError('out_scale')

    # Perform conversion
    in_min = in_scale[0]
    out_min = out_scale[0]
    k = float(out_scale[1] - out_scale[0]) / (in_scale[1] - in_min)
    f = vectorize(lambda x: k * (x - in_min) + out_min, otypes=[IM_FLOAT])
    return f(im)


def sp_read(filename):
    """
    Read an image using SciPy (actually PIL). PIL is faster than ITK saving but does not support MHA.
    Common Supported Formats:
        PNG  (1-bit BW, 8-bit gray, 24-bit RGB)
        BMP  (1-bit BW, 8-bit gray, 24-bit RGB)
        TIFF (1-bit BW, 8-bit gray, 24-bit RGB)
        JPEG (8-bit gray, 24-bit RGB)
        IM   (all?)
    See http://www.pythonware.com/library/pil/handbook/formats.htm for all details
    """
    from scipy.misc import imread
    return imread(filename)
def sp_save(filename, im):
    """
    Save an image using SciPy (actually PIL). PIL is faster than ITK saving but does not support MHA.
    Common Supported Formats:
        PNG  (1-bit BW, 8-bit gray, 24-bit RGB)
        BMP  (1-bit BW, 8-bit gray, 24-bit RGB)
        TIFF (1-bit BW, 8-bit gray, 24-bit RGB)
        JPEG (8-bit gray, 24-bit RGB)
        IM   (all?)
    See thtp://www.pythonware.com/library/pil/handbook/formats.htm for all details
    """
    from scipy.misc import imsave
    if im.dtype == IM_RGB24_STRUCT: im = im.view(dtype=IM_RGB24)
    imsave(filename, im)


def itk_read(filename):
    """
    Read an image using ITK. ITK supports some additional formats but is slower than using SciPy/PIL.
    Common Supported Formats (* means advantage over SciPy/PIL):
        *MHA/MHD (all)*
        *VTK     (all)*
        PNG  (8-bit gray, *16-bit gray*, 24-bit RGB)
        BMP  (8-bit gray, 24-bit RGB)
        TIFF (8-bit gray, *16-bit gray*, 24-bit RGB)
        JPEG (8-bit gray)
    See http://www.paraview.org/Wiki/ITK/File_Formats for more details
    """
    from SimpleITK import GetArrayFromImage, ReadImage
    return GetArrayFromImage(ReadImage(filename))
def itk_save(filename, im):
    """
    Save an image using ITK. ITK supports some additional formats but is slower than using SciPy/PIL.
    Common Supported Formats (* means advantage over SciPy/PIL):
        *MHA/MHD (all)*
        *VTK     (all)*
        PNG  (8-bit gray, *16-bit gray*, 24-bit RGB)
        BMP  (8-bit gray, 24-bit RGB)
        TIFF (8-bit gray, *16-bit gray*, 24-bit RGB)
        JPEG (8-bit gray)
    See http://www.paraview.org/Wiki/ITK/File_Formats for more details
    """
    from SimpleITK import GetImageFromArray, WriteImage
    if im.dtype == IM_RGB24_STRUCT: im = im.view(dtype=IM_RGB24)
    WriteImage(GetImageFromArray(im, isVector=True), filename)
