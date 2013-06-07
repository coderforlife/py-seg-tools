"""
Image utility functions for memseg and related conversion scritps.
"""

from numpy import dtype, int8, uint8, int16, uint16, uint32, float32

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

def is_rgb24(im): return im.ndim == 2 and im.dtype == IM_RGB24_STRUCT or im.ndim == 3 and im.shape[2] == 3 and im.dtype == IM_BYTE
def is_image_besides_rgb24(im): return im.ndim == 2 and im.dtype in (IM_BYTE, IM_USHORT, IM_USHORT_BE, IM_UINT, IM_UINT_BE, IM_SBYTE, IM_SHORT, IM_SHORT_BE, IM_FLOAT)
def is_image(im): return is_rgb24(im) or is_image_besides_rgb24(im)

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
    MAP_ = 0x2050414D
    LITTLE_ENDIAN = 0x00004144
    BIG_ENDIAN = 0x00001717
    # A lot of fields should not be changed! To 'change' them you need to create a new file with the proper data.
    # The following cannot be changed:
    #  nx, ny, mode, next, imodStamp/imodFlags
    # The following should not be changed:
    #  mapc, mapr, maps, nxstart, nystart, nzstart, alpha, beta, gamma [these should always be 1, 2, 3, 0, 0, 0, 90, 90, 90]
    # The following have utility methods to change:
    #  nz, amin, amax, amean, nlabl, mx, my, mz, xlen, ylen, zlen, (cmap, stamp?)
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
    MODIFIABLE_FIELDS = (
     'amin',   'amax',   'amean',
     'ispf',
     'creatid',
     'idtype', 'lens', 'nd1', 'nd2', 'vd1', 'vd2',
     'tiltangles0', 'tiltangles1', 'tiltangles2', 'tiltangles3', 'tiltangles4', 'tiltangles5',
     'xorg',   'yorg',   'zorg',
     'rms',
     'nwave', 'wave1', 'wave2', 'wave3', 'wave4', 'wave5',
     )
    
    def __init__(self, filename, readonly=False, nx=None, ny=None, dtype=None):
        """
        Either opens a previous MRC file or creates a new MRC file.
        In both cases you need to provide a filename.
        When opening an MRC file you can specify if it should be open readonly or not.
        When creating an MRC file you must specify the width, height, and data type (one of the IM_xxx values).
        """
        if nx or ny or dtype:
            ### Creating a new file ###

            # Validate
            if readonly: raise ValueError('readonly')
            self.__dict__['readonly'] = False
            if nx <= 0 or ny <= 0: raise ValueError('nx/ny')

            # Get the mode
            endian = dtype.byteorder
            next = 0
            if dtype == IM_RGB24 or dtype == IM_RGB24_STRUCT: mode = MRC.BYTE_3; dtype = IM_RGB24
            elif dtype == IM_BYTE   or dtype == IM_SBYTE:     mode = MRC.BYTE
            elif dtype == IM_SHORT  or dtype == IM_SHORT_BE:  mode = MRC.SHORT
            elif dtype == IM_USHORT or dtype == IM_USHORT_BE: mode = MRC.USHORT
            elif dtype == IM_FLOAT: mode = MRC.FLOAT
            else: raise ValueError('dtype')

            # Open file (truncates if existing)
            f = open(filename, "w+b")
            self.__dict__['file'] = f

            # Create the header and write it
            h = {
                'nx': nx, 'ny': ny, 'nz': 0,
                'mode': mode,
                'nxstart': 0, 'nystart': 0,'nzstart': 0,
                'mx': nx, 'my': ny, 'mz': 1,
                'xlen': float(nx), 'ylen': float(ny), 'zlen': 1.0,
                'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0, 'mapc': 1, 'mapr': 2, 'maps': 3,
                'amin': 0.0, 'amax': 0.0, 'amean': 0.0,
                'ispf': 0, 'next': 0, 'creatid': 0, 'nint': 0, 'nreal': 0,
                'imodStamp': MRC.IMOD, 'imodFlags': MRC.SIGNED_BYTE_FLAG if dtype == IM_SBYTE else 0,
                'idtype': 0, 'lens': 0, 'nd1': 0, 'nd2': 0, 'vd1': 0, 'vd2': 0,
                'tiltangles0': 0.0, 'tiltangles1': 0.0, 'tiltangles2': 0.0, 'tiltangles3': 0.0, 'tiltangles4': 0.0, 'tiltangles5': 0.0,
                'xorg': 0.0, 'yorg': 0.0, 'zorg': 0.0,
                'cmap': MRC.MAP_, 'stamp': MRC.LITTLE_ENDIAN if endian == '<' else MRC.BIG_ENDIAN,
                'rms': 0.0,
                'nlabl': 1,
                'labels': ['Python MRC Creation'],
            }
            self.__dict__['header'] = h
            self.__write_header(MRC.FIELDS, endian + '10i6f3i3fiih30xhh20xii6h6f3f2ifi')
        else:
            ### Opening an existing file ###
            from struct import unpack

            self.__dict__['readonly'] = bool(readonly)
            f = open(filename, "rb" if readonly else "r+b")
            self.__dict__['file'] = f
            raw = f.read(MRC.HEADER_LEN)

            # Parse Header
            vers = unpack('<ii', raw[208:216])
            endian = '<'
            if vers[0] == MRC.MAP_:
                if vers[1] == MRC.BIG_ENDIAN:
                    endian = '>'
                elif vers[1] != MRC.LITTLE_ENDIAN:
                    raise IOError('MRC file is invalid (stamp is 0x%08x)' % vers[1])
                h = dict(zip(MRC.FIELDS, unpack(endian + '10i6f3i3fiih30xhh20xii6h6f3f2ifi', raw)))
            else:
                h = dict(zip(MRC.FIELDS_OLD, unpack('<10i6f3i3fiih30xhh20xii6h6f6h3fi', raw)))
            self.__dict__['header'] = h

            nx, ny, nz = h['nx'], h['ny'], h['nz']
            mode = h['mode']
            next, nlabl = h['next'], h['nlabl']

            if nx <= 0 or ny <= 0 or nz <= 0:        raise IOError('MRC file is invalid (dims are %dx%dx%d)' % (h['nx'], h['ny'], h['nz']))
            if next < 0:                             raise IOError('MRC file is invalid (extended header size is %d)' % h['next'])
            if nlabl < 0 or nlabl > MRC.LABEL_COUNT: raise IOError('MRC file is invalid (the number of labels is %d)' % h['nlabl'])
            if h['nxstart'] !=  0 or h['nystart'] !=  0 or h['nzstart'] !=  0: raise IOError('MRC file is has an unusual start (%d, %d, %d)'       % (h['nxstart'], h['nystart'], h['nzstart']))
            if h['alpha']   != 90 or h['beta']    != 90 or h['gamma']   != 90: raise IOError('MRC file is has an unusual cell angles (%d, %d, %d)' % (h['alpha'], h['beta'], h['gamma']))
            if h['mapc']    !=  1 or h['mapr']    !=  2 or h['maps']    !=  3: raise IOError('MRC file is has an unusual ordering (%d, %d, %d)'    % (h['mapc'], h['mapr'], h['maps']))

            # TODO: validate mx, my, mz - grid size in X, Y, and Z (are these always equal to nx, ny, nz?)

            # Read labels and extra header data
            h['labels'] = [f.read(MRC.LABEL_LEN) for _ in range(0, nlabl, 1)]
            if next:
                f.seek(MRC.HEADER_LEN + MRC.LABEL_LEN * MRC.LABEL_COUNT)
                h['extra'] = f.read(next)

            # Deterimine data type
            if   mode == MRC.BYTE:   dtype = IM_SBYTE if h['imodStamp'] == MRC.IMOD and h['imodFlags'] & MRC.SIGNED_BYTE_FLAG == MRC.SIGNED_BYTE_FLAG else IM_BYTE
            elif mode == MRC.SHORT:  dtype = IM_SHORT.newbyteorder(endian)
            elif mode == MRC.FLOAT:  dtype = IM_FLOAT
            elif mode == MRC.USHORT: dtype = IM_USHORT.newbyteorder(endian)
            elif mode == MRC.BYTE_3: dtype = IM_RGB24
            elif mode == MRC.SHORT_2 or mode == MRC.FLOAT_2:
                raise IOError('MRC file uses a complex format which is not supported')
            else:
                raise IOError('MRC file is invalid (mode is %d)' % mode)

        # Precompute these for getting sections fast
        self.__dict__['dtype'] = dtype
        self.__dict__['shape'] = (ny, nx) + dtype.shape # does NOT include z dimension
        self.__dict__['data_offset'] = MRC.HEADER_LEN + MRC.LABEL_LEN * MRC.LABEL_COUNT + next
        self.__dict__['stride'] = nx
        self.__dict__['section_size'] = ny * nx
        self.__dict__['section_gap'] = 0
        self.__dict__['section_full_data_size'] = ny * nx * dtype.itemsize

    # General
    def __del__(self): self.close()
    def close(self):
        if self.__dict__['file']:
            self.__dict__['file'].close()
            del self.__dict__['file']
            self.__dict__['file'] = None
    def view(self, x, y, z): return MRCView(self, x, y, z)

    # Forwarding attributes to the header names
    def __getattr__(self, name):
        if name == 'header' or not self.header.has_key(name): raise AttributeError(name)
        return self.header[name]
    def __setattr__(self, name, value):
        if name == 'header' or not self.header.has_key(name) or not name in MRC.MODIFIABLE_FIELDS: raise AttributeError(name)
        self.header[name] = value
    def __dir__(self): return sorted(set(dir(self.__class__) + self.__dict__.keys() + self.header.keys()))
    def pixel_spacing(self, spacing = None):
        """Gets or sets the pixel spacing in the header. Does not write the header to disk."""
        h = self.header
        if spacing == None: return (h['xlen']/h['mx'], h['ylen']/h['my'], h['zlen']/h['mz'])
        if len(spacing) != 3: raise ValueError('spacing')
        h['xlen'] = spacing[0]/h['mx']
        h['ylen'] = spacing[1]/h['my']
        h['zlen'] = spacing[2]/h['mz']
    def __len__(self): return self.header['nz'] # only number of slices

    # Internal section reading and writing
    def _get_section(self, i):
        from numpy import fromfile
        self.file.seek(self.data_offset + i * self.section_full_data_size)
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.shape)
    def _get_next_section(self):
        from numpy import fromfile
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.shape)
    def _set_section(self, i, im):
        if self.dtype == IM_RGB24 and im.ndim == 2 and im.dtype == IM_RGB24_STRUCT: im = im.view(dtype=IM_RGB24)
        if self.shape != im.shape or self.dtype.base != im.dtype: raise ValueError('im')
        self.file.seek(self.data_offset + i * self.section_full_data_size)
        im.tofile(self.file)
    def _set_next_section(self, i, im):
        if self.dtype == IM_RGB24 and im.ndim == 2 and im.dtype == IM_RGB24_STRUCT: im = im.view(dtype=IM_RGB24)
        if self.shape != im.shape or self.dtype.base != im.dtype: raise ValueError('im')
        im.tofile(self.file)

    # Getting Slices
    def __getitem__(self, index):
        nz = self.header['nz']
        if isinstance(index, (int, long)):
            if index < 0: index = nz - index
            if index >= nz: raise KeyError('index')
            return self._get_section(index)
        elif isinstance(index, slice):
            # TODO: should accept lists in addition to slices
            # TODO: slices should return an iterable generator thing instead of images preloaded
            if index.start == None: index.start = 0
            if index.stop == None or index.stop == 2147483647: index.stop = nz
            if index.step == 0 and index.start < index.stop and index.start < 0 or index.start >= nz: raise KeyError('index')
            if index.step == 1 or index.step == None:
                if index.stop > nz: raise KeyError('index')
                if index.start >= index.stop: return []
                l = [self._get_section(index.start)]
                for _ in range(index.start + 1, index.stop): l.append(self._get_next_section())
                return l
            else:
                r = range(index.start, index.stop, index.step)
                if len(r) == 0: return []
                if index.step > 0 and r[-1] >= nz or index.step < 0 and r[-1] < 0: raise KeyError('index')
                l = []
                for z in r: l.append(self._get_section(z))
                return l
        else: raise TypeError('index')
    def __iter__(self):
        yield self._get_section(0)
        for _ in range(1, self.header['nz']): yield self._get_next_section()
    def stack(self):
        from numpy import fromfile
        nz = self.header['nz']
        self.file.seek(self.data_offset)
        return fromfile(self.file, self.dtype, self.section_size * nz).reshape((nz,) + self.shape)

    # Updating/writing header information
    def update_header_pixel_values(self):
        """Updates the header properties 'amin', 'amax', and 'amean' to the current image data. Does not write the header to disk."""
        h = self.header
        nz = h['nz']
        im = self._get_section(0)
        amin = im.min()
        amax = im.max()
        amean = im.mean()
        for _ in range(1, nz):
            im = self._get_next_section()
            amin = min(amin, imin)
            amax = max(amax, imax)
            amean += im.mean()
        h['amin'] = amin
        h['amax'] = amax
        h['amean'] = amean / nz
    def add_label(self, lbl):
        """Adds a label to the header. Does not write the header to disk."""
        if len(lbl) > MRC.LABEL_LEN: raise ValueError('lbl')
        h = self.header
        lbls = h['labels']
        lbls.append(lbl.ljust(MRC.LABEL_LEN))
        if len(lbls) > MRC.LABEL_COUNT: lbls = lbls[-MRC.LABEL_COUNT:]
        h['nlabl'] = len(lbls)
        h['labels'] = lbls
    def write_header(self):
        """Write the header to disk."""
        # Validate and update some header fields
        if self.readonly: raise Exception('readonly')
        h = self.header
        if h['nx'] <= 0 or h['ny'] <= 0 or h['nz'] <= 0: raise ValueError('nx/ny/nz')
        if len(h['labels']) > MRC.LABEL_COUNT: raise ValueError('labels')
        for lbl in h['labels']:
            if len(lbl) > MRC.LABEL_LEN: raise ValueError('label')
        h['nlabl'] = len(h['labels'])
        h['next'] = len(h['extra']) if hasattr(h, 'extra') and h['extra'] else 0

        # Write!
        if hasattr(h, 'cmap'):
            if not hasattr(h, 'stamp') or h['cmap'] != MRC.MAP_ or h['stamp'] != MRC.BIG_ENDIAN and h['stamp'] != MRC.LITTLE_ENDIAN: raise ValueError('cmap/stamp')
            endian = '>' if h['stamp'] == MRC.BIG_ENDIAN else '<'
            self.__write_header(MRC.FIELDS, endian + '10i6f3i3fiih30xhh20xii6h6f3f2ifi')
        else:
            self.__write_header(MRC.FIELDS_OLD, '<10i6f3i3fiih30xhh20xii6h6f6h3fi')
    def __write_header(self, fields, format):
        # Actually write the header (checks must be already done)
        from struct import pack
        h = self.header
        f = self.file
        f.seek(0)
        values = [h[field] for field in fields]
        f.write(pack(format, *values))
        for lbl in h['labels']: f.write(lbl.ljust(MRC.LABEL_LEN))
        blank_lbl = ' ' * MRC.LABEL_LEN
        for _ in xrange(len(h['labels']), MRC.LABEL_COUNT): f.write(blank_lbl)
        if h['next']: f.write(h['extra'])

    # Setting and adding slices        
    def __setitem__(self, index, im):
        """Sets a slice to a new image, writing it to disk. The header values 'amin', 'amax', and 'amean' are not updated since they cannot be accurately updated without reading the replaced slice/all slices."""
        if self.readonly: raise Exception('readonly')
        if not isinstance(index, (int, long)): raise TypeError('index')
        h = self.header
        nz = h['nz']
        if index < 0: index = nz - index
        if index >= nz: raise ValueError('index')
        #old = self._get_section(index)
        #h['amean'] = (h['amean'] * nz - old.mean() + im.mean()) / nz
        #h['amean'] = (h['amean'] * nz + im.mean()) / (nz + 1)
        #h['amin'] = min(h['amin'], im.min())
        #h['amax'] = min(h['amax'], im.max())
        self._set_section(index, im)
    def append(self, im):
        """Appends a single slice, writing it to disk. The header values 'amin', 'amax, 'amean', 'nz', 'mz', and 'zlen' are updated but not written to disk."""
        if self.readonly: raise Exception('readonly')
        h = self.header
        nz = h['nz']
        h['amin'] = min(h['amin'], im.min()) if nz != 0 else im.min()
        h['amax'] = max(h['amax'], im.max()) if nz != 0 else im.max()
        h['amean'] = (h['amean'] * nz + im.mean()) / (nz + 1)
        self._set_section(nz, im)
        nz += 1
        h['nz'] = nz
        h['zlen'] = h['zlen'] / h['mz'] * nz
        h['mz'] = nz
    def append_all(self, ims):
        """Appends many slices, writing them to disk. The header values 'amin', 'amax, 'amean', 'nz', 'mz', and 'zlen' are updated but not written to disk."""
        if self.readonly: raise Exception('readonly')
        ims = iter(ims) # get as an iterator [no-op for something that is already an iterator]
        try: im = ims.next() # first image in iterator
        except StopIteration: return # there are no images to add
        h = self.header
        nz = h['nz']
        amin = min(h['amin'], im.min()) if nz != 0 else im.min()
        amax = max(h['amax'], im.max()) if nz != 0 else im.max()
        amean = h['amean'] * nz + im.mean()
        self._set_section(nz, im)
        nz += 1
        for im in ims:
            # remainder of images in iterator
            amin = min(amin, im.min())
            amax = max(amax, im.max())
            amean += im.mean()
            self._set_next_section(im)
            nz += 1
        h['amin'] = amin
        h['amax'] = amax
        h['amean'] = amean / nz
        h['nz'] = nz
        h['zlen'] = h['zlen'] / h['mz'] * nz
        h['mz'] = nz
    def remove_last_slice(self, count=1):
        """Removes 'count' slices from the end (default 1), shortening the file on disk. The header values 'nz', 'mz', and 'zlen' are updated but not written to disk. The header values 'amin', 'amax, and 'amean' are not updated."""
        if self.readonly: raise Exception('readonly')
        h = self.header
        nz = h['nz']
        if count <= 0 or nz < count: raise ValueError('count')
        nz -= count
        self.file.truncate(self.data_offset + nz * self.section_full_data_size)
        h['nz'] = nz
        h['zlen'] = h['zlen'] / h['mz'] * nz if nz != 0 else 1.0
        h['mz'] = nz if nz != 0 else 1
    def remove_all_slices(self, count=1):
        """Removes all slices, shortening the file on disk. The header values 'amin', 'amax, 'amean', 'nz', 'mz', and 'zlen' are updated but not written to disk."""
        if self.readonly: raise Exception('readonly')
        h = self.header
        self.file.truncate(self.data_offset)
        h['amin'] = 0.0
        h['amax'] = 0.0
        h['amean'] = 0.0
        h['nz'] = 0
        h['zlen'] = 1.0
        h['mz'] = 1

class MRCView(MRC):
    """
    A view in an MRC file is a bit more efficient when cropping in the Y dimension as less data will be read from the file.
    For the X dimension the entire lines are still read but the data is simply ignored.
    The header information is updated appropiately except for the min, max, and mean pixel densities. You can update those using update_header_pixel_values().
    """
    def __init__(self, mrc, x, y, z):
        if not isinstance(mrc, MRC): raise ValueError('mrc')
        mrc_h = mrc.header
        if len(x) != 2 or x[0] < 0 or x[1] < x[0] or x[1] >= mrc_h['nx']: raise ValueError('x')
        if len(y) != 2 or y[0] < 0 or y[1] < y[0] or y[1] >= mrc_h['ny']: raise ValueError('y')
        if len(z) != 2 or z[0] < 0 or z[1] < z[0] or z[1] >= mrc_h['nz']: raise ValueError('z')
        
        x_off, nx = x[0], x[1]-x[0]+1
        y_off, ny = y[0], y[1]-y[0]+1
        z_off, nz = z[0], z[1]-z[0]+1

        self.__dict__['parent'] = mrc
        self.__dict__['readonly'] = True
        self.__dict__['file'] = mrc.file
        h = mrc_h.copy()
        h['nx'] = nx; h['ny'] = ny; h['nz'] = nz
        h['mx'] = nx; h['my'] = ny; h['mz'] = nz
        sx = mrc_h['xlen'] / mrc_h['mx']; sy = mrc_h['ylen'] / mrc_h['my']; sz = mrc_h['zlen'] / mrc_h['mz']
        h['xlen'] = sx * nx; h['ylen'] = sy * ny; h['zlen'] = sz * nz
        h['xorg'] = mrc_h['xorg'] - sx * x_off; h['yorg'] = mrc_h['yorg'] - sy * y_off; h['zorg'] = mrc_h['zorg'] - sz * z_off
        # TODO: add a label? self.add_label('...')
        self.__dict__['header'] = h
        
        dtype = mrc.dtype
        stride = mrc.stride
        sec_full_data_sz = mrc.section_full_data_size

        self.__dict__['dtype'] = dtype
        self.__dict__['shape'] = (ny, nx) + dtype.shape
        self.__dict__['raw_shape'] = (ny, stride) + dtype.shape
        self.__dict__['data_offset'] = mrc.data_offset + x_off + y_off * stride + z_off * sec_full_data_sz
        self.__dict__['stride'] = stride
        self.__dict__['section_size'] = ny * stride
        self.__dict__['section_gap'] = sec_full_data_sz - ny * stride * dtype.itemsize
        self.__dict__['section_full_data_size'] = sec_full_data_sz
        
    def close(self): pass # closing a view is a no-op
    def _get_section(self, i):
        # TODO: this currently wastes memory by reading the entire horizontal axis of the full image then just skipping it in the view (it is still there though and it must all be read)
        from numpy import fromfile
        self.file.seek(self.data_offset + i * self.section_full_data_size)
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.raw_shape)[:,:self.header['nx']]
    def _get_next_section(self):
        # TODO: this currently wastes memory by reading the entire horizontal axis of the full image then just skipping it in the view (it is still there though and it must all be read)
        from numpy import fromfile
        self.file.seek(self.section_gap, 1)
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.raw_shape)[:,:self.header['nx']]
    def stack(self):
        # TODO: check if one fromfile read with a view is more efficient
        from numpy import empty
        nz = self.header['nz']
        self.file.seek(self.data_offset)
        stack = empty((nz,) + self.shape)
        for i, sec in enumerate(self): stack[i,:,:] = sec
        return stack

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
    Order is maintained. Note: Currently signed types where negative values are actually used are not supported.
    """
    # TODO: support using the same numbers across multiple slices
    # TODO: support running connected components code on BW data ( scipy.ndimage.label() )
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
    
    from numpy import empty, multiply, add
    
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
    k = float(out_scale[1] - out_min) / (in_scale[1] - in_min)
    out = empty(im.shape, dtype=IM_FLOAT)
    return add(multiply(im, k, out), out_min - in_min, out)


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
