from collections import Iterable
from gzip import GzipFile
from images import *
from itertools import product
from numpy import array, empty, frombuffer, fromfile, prod, ravel
import re
import os.path

__all__ = ['imread_mha', 'imread_mhd', 'imsave_mha', 'imsave_mhd']

dtype2met = {
    #IM_RGB24  : 'MET_UCHAR_ARRAY', # handle this one specially
    IM_BYTE   : 'MET_UCHAR',
    IM_SBYTE  : 'MET_CHAR',
    IM_SHORT  : 'MET_SHORT',  IM_SHORT_BE  : 'MET_SHORT',
    IM_USHORT : 'MET_USHORT', IM_USHORT_BE : 'MET_USHORT',
    IM_INT    : 'MET_INT',    IM_INT_BE    : 'MET_INT',
    IM_UINT   : 'MET_UINT',   IM_UINT_BE   : 'MET_UINT',
    IM_LONG   : 'MET_LONG',   IM_LONG_BE   : 'MET_LONG',
    IM_ULONG  : 'MET_ULONG',  IM_ULONG_BE  : 'MET_ULONG',
    IM_FLOAT  : 'MET_FLOAT',
    IM_DOUBLE : 'MET_DOUBLE',
}
met2dtype = { v:k for k,v in dtype2met.iteritems() }

file_list = re.compile('^LIST(\s+(\d+)D?)?$', re.IGNORECASE)
file_pattern = re.compile('^(\S*%[#0-9 +.-]*[hlL]?[dD]\S*)\s+(\d+)\s+(\d+)\s+(\d+)$')

def _split(x):
    for c in ',;:':
        if c in x: return x.split(c)
    return x.split()
def _bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, basestring):
        if x.lower() == 'false' or x.lower() == 'f': return False
        if x.lower() == 'true'  or x.lower() == 't': return True
    return bool(x)

####### Reading ################################################################

def _fromfile(f, dtype, npxls, compressed):
    if compressed:
        with GzipFile(f, type='zlib') as gz: return frombuffer(gz.read(), dtype, npxls) # could use off directly here
    return fromfile(f, dtype, npxls)
def read_data_file(datafile, off, compressed, dtype, shape):
    npxls = prod(shape)
    off = os.path.getsize(datafile) - npxls * dtype.itemsize if off == -1 else off
    if off < 0: raise ValueError('MHA/MHD file data file is not large enough for the pixel data')
    if off == 0 and not compressed: return fromfile(f, dtype, npxls).reshape(shape)
    with open(datafile, 'rb') as f:
        f.seek(off)
        return _fromfile(f, dtype, npxls, compressed).reshape(shape)
        #return fromfile(GzipFile(f, type='zlib') if compressed else f, dtype, npxls).reshape(shape)
def re_search(re, string):
    re_search.last_match = m = re.search(string)
    return m != None

def imread_mha(filename):
    """Equivilent to imread_mhd, MHD vs MHA files are determined based on the header in the file"""
    return imread_mhd(filename)
def imread_mhd(filename):
    """
    Read an MHA/MDA image file.

    Supported image data formats:
        Single channel unsigned and signed integral data types up to 8 bytes each
        Single channel float and double
        3-channel unsigned byte (MET_UCHAR_ARRAY[3])

    Unsupported features:
        Non-image files
        ASCII files
        HeaderSize of -1 when data is compressed (this is probably invalid anyways)
        Use of ElementByteOrderMSB (unknown purpose)
        Many image data formats
        Many fields are simply ignored (e.g. TransformationMatrix) but they are returned
    """
    directory = os.path.dirname(os.path.realpath(filename))

    # Read Header
    tags_raw = {}
    tags = {}
    files = None
    with open(filename, 'rb') as f:
        line = f.readline(256)
        while len(line) > 0 and re.search('^[A-Za-z0-9_-]+\s*=', line) != None:
            if line[-1] != '\n': line += f.readline() # read the rest of the line
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            k_ = k.lower()
            tags_raw[k] = v
            tags[k_] = v
            if k_ == 'objecttype' and v.lower() != 'image': raise ValueError('Non-image MHA/MHD files are not supported')
            if k_ == 'elementdatafile' and re_search(file_list, v):
                # we have a list of files
                file_ndims = int(re_search.last_match.group(2)) if re_search.last_match.lastindex == 1 else int(tags['ndims']) - 1
                nfiles = prod([int(x) for x in _split(tags['dimsize'])][file_ndims:])
                files = [] # used later
                for i in xrange(nfiles): files.append(f.readline())
            line = f.readline(256)
        off = f.tell() - len(line)

    # Check/Parse Header
    for k in ('ObjectType', 'NDims', 'DimSize', 'ElementType', 'ElementDataFile'):
        if k.lower() not in tags: raise ValueError('MHA/MHD file header does not have required field \''+k+'\'')
    #if tags['objecttype'].lower() != 'image': raise ValueError('Non-image MHA/MHD files are not supported')
    ndims = int(tags['ndims'])
    if ndims <= 0: raise ValueError('Invalid number of dimensions in MHA/MHD file')
    shape = [int(x) for x in _split(tags['dimsize'])]
    if len(shape) != ndims or any(x <= 0 for x in shape): raise ValueError('Invalid dimension sizes in MHA/MHD file')
    if ndims >= 2: shape[0], shape[1] = shape[1], shape[0]
    if not _bool(tags.get('binarydata', True)): raise ValueError('ASCII MHA/MHD files not supported')
    compressed = _bool(tags.get('compresseddata', False))
    headersize = int(tags.get('headersize', 0))
    if headersize < -1: raise ValueError('MHA/MHD HeaderSize is invalid')
    if compressed and headersize == -1: raise ValueError('MHA/MHD HeaderSize cannot be -1 when CompressedData is True')

    # Check/Parse Element Type Header Keys
    elemtype = tags['elementtype'].upper()
    # TODO: use ElementByteOrderMSB?
    endian = '>' if _bool(tags.get('binarydatabyteordermsb', False)) else '<'
    nchannels = int(tags.get('elementnumberofchannels', 1))
    if nchannels == 3:
        if elemtype != 'MET_UCHAR_ARRAY': raise ValueError('MHA/MHD file image type not supported')
        dtype = IM_RGB24
    elif nchannels == 1:
        if elemtype not in met2dtype: raise ValueError('MHA/MHD file image type not supported')
        dtype = met2dtype[elemtype].newbyteorder(endian)
    else:
        raise ValueError('MHA/MHD file image type not supported')

    # TODO: make these values translatable into MRC fields:
    #ElementMin, ElementMax
    #ElementSpacing, ElementSize
    #Offset/Position/Origin, CenterOfRotation, Rotation/Orientation/TransformMatrix

    # Read data
    datafile = tags['elementdatafile']
    if datafile.upper() == 'LOCAL':
        # read data from the given file, starting at the offset
        return tags_raw, read_data_file(filename, off, compressed, dtype, shape)
    elif re_search(file_list, datafile):
        # a list of files
        file_ndims = int(re_search.last_match.group(2)) if re_search.last_match.lastindex == 1 else ndims - 1
        data = empty(shape, dtype)
        shape = shape[:file_ndims]
        ind = [slice(None)] * ndims
        inds = product(*(xrange(x) for x in shape[file_ndims:]))
        for i, f in enumerate(files):
            ind[file_ndims:] = inds.next()
            data[ind] = read_data_file(os.path.realpath(os.path.join(directory, f)), headersize, compressed, dtype, shape)
        return tags_raw, data
    elif re_search(file_pattern, datafile):
        # follow a pattern to find all the files to be loaded
        pattern, start, stop, step = re_search.last_match.groups(0)
        start, stop, step = int(start), int(stop), int(step)
        if stop < start or step <= 0 or shape[-1] != (stop - start)//step + 1: raise ValueError('MHA/MHD invalid ElementDataFile')
        data = empty(shape, dtype)
        shape = shape[:-1]
        for i in xrange(start, stop + 1, step):
            data[..., i] = read_data_file(os.path.realpath(os.path.join(directory, pattern % i)), headersize, compressed, dtype, shape)
        return tags_raw, data
    else:
        # datafile is just a file, starting at headersize
        return tags_raw, read_data_file(datafile, headersize, compressed, dtype, shape)

####### Saving #################################################################

def list2str(l): return ' '.join(str(x) for x in l)
def lookup(l):   return {x.lower():x for x in l}
def float_list(k, v, n): # key, value, number of floats
    if isinstance(v, basestring): v = _split(v)
    if isinstance(v, Iterable):
        v = [float(V) for V in ravel(v)] # ravel is necessary for multidimensional inputs, particularly for Orientation values
        if len(v) == 1: v *= n
        elif len(v) != n: raise ValueError('Wrong number of values for "' + k + '"')
    else: v = [float(v)] * n
    return list2str(v)
def tofile(im, f, compressed):
    if compressed:
        from numpy import array_split
        with GzipFile(f, 'wb', 9, 'zlib') as gz:
            for chunk in array_split(im.ravel(), max(im.size // 10485760, 1)): # 10 MB chunks
                gz.write(buffer(chunk))
    else: im.tofile(f)

def imsave_mha(filename, im, CompressedData=False, **tags):
    """Save an image as an MHA image (data embeded in the metadata file). See imsave_mhd for more information."""
    imsave_mhd(filename, im, 'LOCAL', CompressedData, **tags)
def imsave_mhd(filename, im, datafile=None, CompressedData=False, **tags):
    """
    Save an image as an MHD image.
    
    If the datafile name is not given it will be automatically generated. If it is LOCAL than no
    seperate file will be created. The datafile cannot be 'LIST' or start with 'LIST #'.

    If you set CompressedData to True it will cause the image data to be compressed. This will slow
    down saving but in many cases will result in significant diskspace savings.
    
    You may specify extra tags to be saved in the image header. These are ignored for the most part
    and simply copied into the header. Known tags are checked and possibly corrected. For any tag
    that requires one value per dimension and only a single value is provided it will automatically
    be copied for each dimension.

    Currently many features of MHA/MHD files are not supported, including:
        Not all data types are understood
        Outputing data to multiple datafiles
        ASCII data
        Non-image objects
        Creating a datafile with a non-MHA/MHD header (HeaderSize)
    Attempting to force these features through tags will raise errors.
    """    
    if im.dtype == IM_RGB24_STRUCT: im = im.view(dtype=IM_RGB24)

    ndims = im.ndim
    filename = os.path.realpath(filename.strip())
    directory = os.path.dirname(filename)
    if datafile == None:
        datafile = filename + '.dat'
        if os.path.exists(datafile):
            from tempfile import mktemp
            datafile = mktemp('.dat', os.path.basename(filename), directory)
    else: datafile = datafile.strip()
    if datafile.upper() == 'LOCAL': datafile = 'LOCAL'
    elif file_list.search(datafile) != None or file_pattern.search(datafile) != None:
        # TODO: support ElementDataFile = LIST [XD] or %03d min max step
        raise ValueError('Datafile name "'+datafile+'" is a reserved name that isn\'t supported')

    # These tags are not allowed in the extra tags as they are either implied from the image data or have required values
    # TODO: support BinaryData = False
    # TODO: support having the not_allowed entries but only if they are exactly what we would make them to be
    not_allowed = lookup(('ObjectType', 'ObjectSubType',
                   'NDims', 'DimSize',
                   'BinaryData', 'BinaryDataByteOrderMSB', 'ElementByteOrderMSB' # what is difference here?
                   'CompressedData', # special parameter
                   'HeaderSize', # used to specify how data is stored, int >= -1 with -1 as special
                   'ElementType', 'ElementNumberOfChannels', 'ElementDataFile'))
    string    = lookup(('Name', 'Comment', 'TransformType', 'Modality')) # TODO: TransformType and Modality are actually enums
    ints      = lookup(('ID', 'ParentID'))
    sames     = lookup(('ElementMin', 'ElementMax')) # values that must be the same as the data values
    n_floats  = lookup(('ElementSize', 'ElementSpacing', 'Offset', 'Position', 'Origin', 'CenterOfRotation'))
    n2_floats = lookup(('Rotation', 'Orientation', 'TransformMatrix'))

    # These are synonyms, only one of each is allowed
    only_one = (
        ('Offset', 'Position', 'Origin'),
        ('Rotation', 'Orientation', 'TransformMatrix'),
        )

    for k_,v in tags.items():
        k = k_.lower()
        if k in not_allowed: raise ValueError('Setting the tag "' + k_ + '" is not allowed')
        elif k in string:
            k = string[k]
            v = str(v)
            if '\n' in v: raise ValueError('Invalid value "' + v + '" for tag "' + k_ + '"')
        elif k in ints:
            k = ints[k]
            v = str(int(v))
        elif k in sames:
            k = sames[k]
            a = array(v,im.dtype.base)
            if a.shape != im.dtype.shape: raise ValueError('Invalid value "' + v + '" for tag "' + k_ + '"')
            v = str(a).strip('[]')
        elif k in n_floats:
            k = n_floats[k]
            v = float_list(k, v, ndims)
        elif k in n2_floats:
            k = n2_floats[k]
            v = float_list(k, v, ndims * ndims)
        elif k == 'color':
            k = 'Color'
            v = float_list(k, v, 4)
        # TODO:
        #elif k == 'sequenceid':
        #    k = 'SequenceID'
        #    ...
        elif k == 'anatomicalorientation':
            # string ndims chars long, each has to be [R|L] | [A|P] | [S|I] and form a distinct set, can be ? for unknown
            k = 'AnatomicalOrientation'
            v = ''.join(v)
            v = v.strip().upper()
            if len(v) != ndims: raise ValueError('AnatomicalOrientation tag must have one letter per dimension')
            options = { 'R' : 'L', 'L' : 'R', 'A' : 'P', 'P' : 'A', 'S' : 'I', 'I' : 'S' }
            for x in v:
                if x == '?': continue
                if x not in options: raise ValueError('AnatomicalOrientation tag is not well-formed')
                del options[options[x]]
                del options[x]
        elif re.search('[^a-z0-9_-]', k) != None:
            raise ValueError('Invalid key name "' + k_ + '"')
        else: continue
        del tags[k_]
        tags[k] = v
    for name_set in only_one:
        found_one = False
        for name in name_set:
            if name in tags:
                if found_one: raise ValueError('There can only be one tag from the set '+(', '.join(str(x) for x in only_one)))
                else: found_one = True

    shape_x = list(im.shape)
    if ndims >= 2: shape_x[0], shape_x[1] = shape_x[1], shape_x[0]
    alltags = [
        ('ObjectType', 'Image'),
        ('NDims', str(ndims)),
        ('DimSize', list2str(shape_x)),
        ('BinaryData', 'True'),
        ('BinaryDataByteOrderMSB', str(dtype.byteorder == '>' or dtype.byteorder == '=' and sys.byteorder != 'little')),
        ('CompressedData', str(_bool(CompressedData))),
        ]
    alltags.extend(tags.iteritems())
    if is_rgb24(im):
        alltags.append(('ElementType', 'MET_UCHAR_ARRAY'))
        alltags.append(('ElementNumberOfChannels', '3'))
    else:
        if im.dtype not in dtype2met: raise ValueError('Format of image is not supported')
        alltags.append(('ElementType', dtype2met[im.dtype]))
    alltags.append(('ElementDataFile', os.path.relpath(datafile, directory)))

    with open(filename, 'wb') as f:
        for name, value in alltags: f.write(name + ' = ' + str(value) + '\n')
        if datafile == 'LOCAL': tofile(im, f, CompressedData)
    if datafile != 'LOCAL': tofile(im, datafile, CompressedData)

tags, im = imread_mha('0000.mha')
tags_out = {'CompressedData':True,'AnatomicalOrientation':'??','Offset':(0,0),'TransformMatrix':(1,0,0,1),'CenterOfRotation':(0,0)}
imsave_mha('out.mha', im, **tags_out)
imsave_mhd('out.mhd', im, **tags_out)

import pylab as plt

tags, im = imread_mhd('out.mha')
plt.figure()
plt.imshow(im)
plt.gray()

tags, im = imread_mhd('out.mhd')
plt.figure()
plt.imshow(im)
plt.gray()

plt.show()
