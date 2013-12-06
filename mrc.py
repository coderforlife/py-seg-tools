from numpy import fromfile
from images import IM_BYTE, IM_SBYTE, IM_SHORT, IM_SHORT_BE, IM_USHORT, IM_USHORT_BE, IM_FLOAT, IM_RGB24, IM_RGB24_STRUCT

__all__ = ['MRC']

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

    This code does not support complex formats (typically when saving Fourier-space images).
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

    @staticmethod
    def __get_endian(dtype):
        endian = dtype.byteorder
        if endian == '|': return '<' # | means N/A (single byte), report as little-endian
        elif endian == '=': # is native byte-order
            from sys import byteorder # get the native byte order as 'little' or 'big'
            return '<' if byteorder == 'little' else '>'
        return endian
    
    def __init__(self, filename, readonly=False, nx=None, ny=None, dtype=None):
        """
        Either opens a previous MRC file or creates a new MRC file.
        In both cases you need to provide a filename.
        When opening an MRC file you can specify if it should be open readonly or not.
        When creating an MRC file you must specify the width, height, and data type (one of the IM_xxx values).
        """
        from struct import unpack
        
        if nx or ny or dtype:
            ### Creating a new file ###

            # Validate
            if readonly: raise ValueError('readonly')
            self.__dict__['readonly'] = False
            if nx <= 0 or ny <= 0: raise ValueError('nx/ny')

            # Get the mode
            endian = MRC.__get_endian(dtype)
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
            self.__dict__['readonly'] = bool(readonly)
            f = open(filename, 'rb' if readonly else 'r+b')
            self.__dict__['file'] = f
            raw = f.read(MRC.HEADER_LEN)

            # Parse Header
            vers = unpack('<ii', raw[208:216])
            endian = '<'
            if vers[0] == MRC.MAP_:
                en = vers[1]
                en_1 = en & 0xFF
                en_432 = en & 0xFFFFFF00
                if en == MRC.BIG_ENDIAN or (en_1 == (MRC.BIG_ENDIAN & 0xFF) and en_432 == 0):
                    endian = '>'
                elif en != MRC.LITTLE_ENDIAN and (en_1 != (MRC.LITTLE_ENDIAN & 0xFF) or en_432 != 0):
                    f.close()
                    raise IOError('MRC file is invalid (stamp is 0x%08x)' % en)
                h = dict(zip(MRC.FIELDS, unpack(endian + '10i6f3i3fiih30xhh20xii6h6f3f2ifi', raw)))
            else:
                h = dict(zip(MRC.FIELDS_OLD, unpack('<10i6f3i3fiih30xhh20xii6h6f6h3fi', raw)))
            self.__dict__['header'] = h

            nx, ny, nz = h['nx'], h['ny'], h['nz']
            mode = h['mode']
            next, nlabl = h['next'], h['nlabl']

            if nx <= 0 or ny <= 0 or nz <= 0:        f.close(); raise IOError('MRC file is invalid (dims are %dx%dx%d)' % (h['nx'], h['ny'], h['nz']))
            if next < 0:                             f.close(); raise IOError('MRC file is invalid (extended header size is %d)' % h['next'])
            if not (0 <= nlabl <= MRC.LABEL_COUNT):  f.close(); raise IOError('MRC file is invalid (the number of labels is %d)' % h['nlabl'])
            if h['nxstart'] !=  0 or h['nystart'] !=  0 or h['nzstart'] !=  0: f.close(); raise IOError('MRC file is has an unusual start (%d, %d, %d)'       % (h['nxstart'], h['nystart'], h['nzstart']))
            if h['alpha']   != 90 or h['beta']    != 90 or h['gamma']   != 90: f.close(); raise IOError('MRC file is has an unusual cell angles (%d, %d, %d)' % (h['alpha'], h['beta'], h['gamma']))
            if h['mapc']    !=  1 or h['mapr']    !=  2 or h['maps']    !=  3: f.close(); raise IOError('MRC file is has an unusual ordering (%d, %d, %d)'    % (h['mapc'], h['mapr'], h['maps']))

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
                f.close()
                raise IOError('MRC file uses a complex format which is not supported')
            else:
                f.close()
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
    @property
    def pixel_spacing(self):
        """Gets the pixel spacing of the data"""
        h = self.header
        return (h['xlen']/h['mx'], h['ylen']/h['my'], h['zlen']/h['mz'])
    @pixel_spacing.setter
    def pixel_spacing(self, value):
        """Sets the pixel spacing in the header but does not write the header to disk."""
        if len(spacing) != 3: raise ValueError()
        h = self.header
        h['xlen'] = spacing[0]/h['mx']
        h['ylen'] = spacing[1]/h['my']
        h['zlen'] = spacing[2]/h['mz']
    def __len__(self): return self.header['nz'] # only number of slices

    # Internal section reading and writing
    def _get_section(self, i):
        self.file.seek(self.data_offset + i * self.section_full_data_size)
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.shape)
    def _get_next_section(self):
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.shape)
    def _set_section(self, i, im):
        if self.dtype == IM_RGB24 and im.ndim == 2 and im.dtype == IM_RGB24_STRUCT: im = im.view(dtype=IM_RGB24)
        if self.shape != im.shape or self.dtype.base != im.dtype: raise ValueError('im')
        self.file.seek(self.data_offset + i * self.section_full_data_size)
        im.tofile(self.file)
    def _set_next_section(self, im):
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
    @property
    def stack(self):
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
        h['next'] = len(h['extra']) if 'extra' in h and h['extra'] else 0

        # Write!
        if 'cmap' in h:
            if 'stamp' not in h or h['cmap'] != MRC.MAP_ or h['stamp'] != MRC.BIG_ENDIAN and h['stamp'] != MRC.LITTLE_ENDIAN: raise ValueError('cmap/stamp')
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
        #h['amean'] = (h['amean'] * nz - old.mean() + im.mean()) / nz # inefficient
        #h['amean'] = (h['amean'] * nz + im.mean()) / (nz + 1) # inaccurate
        #h['amin'] = min(h['amin'], im.min()) # inaccurate
        #h['amax'] = min(h['amax'], im.max()) # inaccurate
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
        self.file.seek(self.data_offset + i * self.section_full_data_size)
        return fromfile(self.file, self.dtype, self.section_size).reshape(self.raw_shape)[:,:self.header['nx']]
    def _get_next_section(self):
        # TODO: this currently wastes memory by reading the entire horizontal axis of the full image then just skipping it in the view (it is still there though and it must all be read)
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
