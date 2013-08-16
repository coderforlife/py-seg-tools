"""
Functions that read and write gzipped files.

The user of the file doesn't have to worry about the compression, but random access is not allowed.

The Python gzip module was used for some inspiration, particularly with reading files (read and
readline are nearly verbatim from it).

Additions over the default Python gzip module:
    * Supports pure deflate and zlib data in addition to gzip files
    * Supports modifying and retrieving all the gzip header properties
    * Adds and checks header checksums for increased file integrity
    * Does not use seek except for rewind and explicit negative seeks (means you can use it on
      unbuffered socket connections when not using those features)
    * Adds utility functions for compressing and decompressing buffers and files in the different
      formats. Can guess which of the three formats a file may be in.

Breaking changes:
    * Does not add an 'open()' function (but would be trivial to add)
    * The constructor has changed:
        * 'filename' and 'fileobj' have been combined into a single argument 'output'
        * 'compresslevel' is now just 'level'
        * fourth argument is now 'type' for the type of output ('deflate', 'zlib', 'gzip')
        * 'mtime' is now only supported as a keyword argument when writing gzip files
        * to include a filename in the gzip header, provide it as a keyword argument
        * Overall: if using 3 or less non-keyword arguments it will work as before otherwise not
    * Undocumented properties have essentially all been removed or renamed, most notably:
        * 'fileobj' and 'myfileobj' are now 'base' ('owns_handle' determines if it is 'my' or not)
        * 'mode' is the actual file mode instead of a 1 or 2 indicating READ or WRITE
        * 'mtime' is now 'gzip_options['mtime']' when type is 'gzip' (otherwise not available)
        * deprecated 'filename' is now 'gzip_options['filename']' when type is 'gzip' (otherwise not available)
"""


from os import name as os_name
from sys import maxint
from struct import pack, unpack
from time import time
from io import BufferedIOBase

from zlib import adler32, crc32
from zlib import DEFLATED, MAX_WBITS
from zlib import compress as zcompress, compressobj, Z_FINISH, Z_SYNC_FLUSH, Z_FULL_FLUSH
from zlib import decompress as zdecompress, decompressobj

__all__ = ['gzip_oses', 'default_gzip_os',
           'compress_file', 'decompress_file', 'compress', 'decompress',
           'guessfiletype', 'guesstype',
           'GzipFile']

FTEXT, FHCRC, FEXTRA, FNAME, FCOMMENT = 0x01, 0x02, 0x04, 0x08, 0x10

def no_checksum(data, value=None): return 0
checksums = {
        'gzip' : crc32,
        'zlib' : adler32,
        'deflate' : no_checksum,
    }
gzip_oses = {
        'FAT' : 0,
        'Amiga' : 1,
        'VMS' : 2,
        'Unix' : 3,
        'VM/CMS' : 4,
        'Atari TOS' : 5,
        'HPFS' : 6,
        'Macintosh' : 7,
        'Z-System' : 8,
        'CP/M' : 9,
        'TOPS-20' : 10,
        'NTFS' : 11,
        'QDOS' : 12,
        'Acorn RISCOS' : 13,
        'Unknown' : 255,
    }
default_gzip_oses = {
        'nt' : 0, 'os2' : 0, 'ce' : 0, # FAT (NT could also be NTFS or HPFS and OS/2 could be HPFS)
        'posix' : 3, # UNIX
        'riscos' : 13, # Acorn RISCOS
    }
default_gzip_os = default_gzip_oses.get(os_name, 255) # default is unknown, including for 'java'

def get_filename(f, default=None):
    if isinstance(f, basestring):
        return f
    elif hasattr(f, 'name') and (len(f.name) < 2 or f.name[0] != '<' and f.name[-1] != '>'):
        return f.name
    return default
def gzip_header_str(s):
    if not s: return None
    i = s.find('\x00')
    if i >= 0: s = s[:i]
    s = s.encode('iso-8859-1')
    return s + '\x00' if s else None
def write_gzip_header_str(file, s, chk16):
    file.write(s)
    return crc32(s, chk16) & 0xffffffffL
def read_gzip_header_str(read, chk16):
    s = ''
    while True:
        c = read(1)
        if not c or c == '\x00': break
        s += c
    return s.decode('iso-8859-1'), (crc32(s+'\x00', chk16) & 0xffffffffL)


# GZIP Format {Little-Endian}
# 1F 8B CM FG [MTIME (4)] XF OS
#   CM = 08 - deflate compression method
#   FG = 01 - file is an ASCII text file
#        02 - CRC16 for header is present
#        04 - extra fields are present
#        08 - original file name is present
#        10 - comment is present
#   MTIME = mod. time as secs since 00:00:00 GMT 01/01/70 of the orig file, when compression started, or 0
#   XF = 2 for max compression, 4 for fastest compression
#   OS = the filesystem where the file came from
# [extra data]
# [filename]
# [comment]
# CRC16 checksum of header
# <compressed data>
# CRC32 checksum
# Size of uncompressed data
# 
# ZLIB Format {Big-Endian}
# CM-CF FL [DICT (4)]
#   CM = bits 0 to 3 => 8 for deflate compression method
#   CF = bits 4 to 7 => base-2 logarithm of the LZ77 window size minus 8 (0 is 256, 7 is 32K, 8 and above are not allowed)
#   FL = bits 0 to 4 => check bits for CM-CF, when viewed as a 16-bit int is a multiple of 31 (CMCF*256 + FL)
#        bit  5      => 0 almost exclusively, 1 if there is a dictionary to be used (which won't be supported)
#        bits 6 to 7 => compression level, 0 is fastest => 3 is slowest/max
#   DICT = not supported
# <compressed data>
# Adler32 checksum

def compress_file(input, output=None, level=9, type=None):
    # Get output filename
    in_filename = get_filename(input)
    if output == None:
        if in_filename == None: raise ValueError('Unable to determine output filename')
        output = in_filename + ('.gz' if type=='gzip' or type==None else ('.zlib' if type=='zlib' else '.deflate'))

    # Get gzip options
    opts = {}
    if type == 'gzip' or type == None:
        import os
        try_fstat = True
        if in_filename:
            opts['filename'] = os.path.basename(in_filename) 
            try:
                opts['mtime'] = os.path.getmtime(in_filename)
                try_fstat = False
            except: pass
        if try_fstat:
            try:
                opts['mtime'] = os.fstat(input.fileno()).st_mtime
            except: pass

    # Copy data
    with GzipFile(output, 'wb', level, type, **opts) as output:
        owns_handle = isinstance(input, basestring)
        if owns_handle: input = open(input, 'rb')
        try:
            while True:
                buf = input.read(10*1024*1024)
                if len(buf) == 0: break
                output.write(buf)
        finally:
            if owns_handle: input.close()

def decompress_file(input, output=None, type=None):
    with GzipFile(input, 'rb', type=type) as input:
        # Get the output filename if not provided
        in_filename = get_filename(input)
        if not output:
            if input.type == 'gzip':
                output = input.gzip_options.get('filename')
                if not output and in_filename and in_filename.endswith('.gz'):
                    output = in_filename[:-3]
            elif input.type == 'zlib' and in_filename and in_filename.endswith('.zlib'):
                output = in_filename[:-5]
            elif input.type == 'deflate' and in_filename and in_filename.endswith('.deflate'):
                output = in_filename[:-8]
            if not output: raise ValueError('Unable to determine output filename')

        # Copy data
        owns_handle = isinstance(output, basestring)
        if owns_handle: output = open(output, 'wb')
        try:
            while True:
                buf = input.read(10*1024*1024)
                if len(buf) == 0: break
                output.write(buf)
        finally:
            if owns_handle: output.close()

        # Set mtime on output file if it is available
        if in_filename and input.type == 'gzip' and input.gzip_options['mtime']:
            import os

def compress(input, level=9, type=None):
    level = int(level)
    if type == 'gzip' or type == None:
        xf = 2 if level >= 7 else (4 if level <= 2 else 0)
        s = b'\x1F\x8B\x08\x02' + pack('<LB', int(time()), xf) + b'\xFF'
        s += pack('<H', crc32(s) & 0xffff)
        s += zcompress(input, level)
        s += pack('<LL', crc32(input) & 0xffffffffL, len(input) & 0xffffffffL)
        return s
    elif type == 'zlib':
        header = 0x7800 + (((level+1)//3) << 6)
        mod31 = header % 31
        if mod31 != 0: header += (31 - mod31)
        s += pack('>H', header)
        s += zcompress(input, level)
        s += pack('<L', adler32(input) & 0xffffffffL)
        return s
    elif type == 'deflate':
        return zcompress(input, level)
    else:
        raise ValueError('Compression type must be one of deflate, zlib, or gzip')

def decompress(input, type=None):
    if type == None: type = guesstype(input)
    if type == 'gzip':
        magic1, magic2, method, flags, mtime, xf, os = unpack('<BBBBIBB', input[:10])
        if magic1 != 0x1F or magic2 != 0x8B: raise IOError('Not a gzipped file')
        if method != 8: raise IOError('Unknown compression method')
        if flags & 0xE0: raise IOError('Unknown flags')
        off = unpack('<H', input[10:12])[0] + 12 if flags & FEXTRA else 10
        if flag & FNAME:    off = input.index('\x00', off) + 1
        if flag & FCOMMENT: off = input.index('\x00', off) + 1
        if flags & FHCRC:
            if unpack('<H', input[off:off+2])[0] != (crc32(input[:off]) & 0xffff): raise IOError('Header corrupted')
            off += 2
        crc32, isize = unpack('<II', input[-8:])
        s = zdecompress(input[off:-8], -MAX_WBITS, isize)
        checksum = crc32(s)
        if crc32 != checksum: raise IOError("CRC32 check failed %08x != %08x" % (crc32, checksum))
        if isize != (len(s) & 0xffffffffL): raise IOError("Incorrect length of data produced")
        return s
    elif type == 'zlib':
        header = unpack('>H', input[:2])[0]
        method = (header >>  8) & 0xF
        windowsize = (header >> 12) & 0xF
        fdict  = (header & 0x20) != 0
        if method != 8 or window == 8 or fdict: raise IOError('Unknown compression method')
        if header % 31 != 0: raise IOError('Header corrupted')
        s = zdecompress(input[2:-4], -windowsize)
        a32 = unpack('>I', input[-4:])[0]
        checksum = adler32(s)
        if a32 != checksum: raise IOError("Adler32 check failed %08x != %08x" % (a32, checksum))
        return s
    elif type == 'deflate':
        return zdecompress(input)
    else:
        raise ValueError('Compression type must be one of deflate, zlib, gzip, or None')

def guessfiletype(f):
    if isinstance(f, basestring):
        with open(f, 'rb') as f: return guesstype(f.read(3))
    else: return guesstype(f.read(3))

def guesstype(buf):
    if len(buf) > 2 and ord(buf[0]) == 0x1F and ord(buf[1]) == 0x8B and ord(buf[2]) == 0x08: return 'gzip' # could also check flags and checksum, but this seems good enough
    elif len(buf) > 1 and (ord(buf[0]) & 0xF) == 0x8 and ((ord(buf[0]) >> 4) & 0xF) <= 0x7 and (ord(buf[0]) * 0xFF + ord(buf[1])) % 31 == 0: return 'zlib' # about a 1/1000 chance of guessing zlib when actually deflate
    else: return 'deflate'

class GzipFile(BufferedIOBase):
    offset = 0
    max_read_chunk = 10 * 1024 * 1024 # 10Mb

    def __init__(self, output, mode=None, level=9, type=None, **kwargs):
        """
        Creates a file-like object that wraps another file-like object and either compresses all
        data written to it or decompresses all data read from it.

        If you are outputing to a file you can provide the filename as output. You can also provide
        file-like objects for output that follow these rules:
            If writing compressed data it must have:
                mode property / attribute if not provided to the constructor
                write amd flush methods
            If reading compressed data it must have:
                read method
            If you want to rewind or negative seek:
                seek method
            Optionally used if available:
                name property / attribute
                fileno and isatty methods
        If you provide a file-like object it will not be closed when this object is closed. This can
        be changed by setting the property owns_handle to True.

        Mode should be 'rb' (default) for reading or 'wb' or 'ab' for writing. The 'b' can be left
        off. The letters t, U and + are all ignored. If writing to a file-like object that has a
        mode property or attribute you do not need to give the mode here.

        The type must be 'deflate' (RFC 1951), 'zlib' (RFC 1950), or 'gzip' (RFC 1952). Default is
        None which means 'gzip' for writing and guess the format when reading. However, it is always
        best if you know the format to supply it.

        The level is the compression level from 0 (none) to 9 (maximum compression - default). It is
        ignored when reading.

        When writing gzip data you can include extra information with the following keyword arguments:
            os= to an integer from 0 to 255 that describes the filesystem where the file orginated (default depends on system)
            mtime= to an integer representing the modification time of the original file as a UNIX timestamp (default is now)
            text=True if the data being written is text (default is binary)
            filename= to the original filename that is being compressed
            comment= to a user-readable comment
            extras= to a list of 2-element tuples, each has a 2 byte string for the subfield id and a byte string for the subfield data

        When reading gzip data the extra information is available from the gzip_options property.
        """

        # Check mode
        if not mode: mode = output.mode if not isinstance(output, basestring) and hasattr(output, 'mode') else 'rb'
        mode = str(mode).translate(None, 'Ut+') # remove unsupported mode characters
        if mode[0] == 'r': mode = 'rb'
        elif mode[0] == 'w': mode = 'wb'
        elif mode[0] == 'a': mode = 'ab'
        else: raise 'Mode ' + mode + ' not supported'

        # Check type
        if type not in ('deflate', 'zlib', 'gzip', None): raise ValueError('Compression type must be one of deflate, zlib, or gzip (or None if reading)')
        if mode[0] != 'r' and type == None: type = 'gzip'

        # Check level
        level = int(level)
        if level < 0 or level > 9: raise ValueError('Compression level must be between 0 and 9 (inclusive)')

        # Check kwargs
        if kwargs and (type != 'gzip' or mode[0] == 'r'): raise ValueError('Extra keyword arguments can only be provided when writing gzip data')
        if type == 'gzip' and mode[0] != 'r':
            if len(kwargs.viewkeys() - {'text', 'os', 'comment', 'filename', 'mtime', 'extras'}): raise ValueError('Gzip settings must only include text, comment, filename, mtime, and extras')
            is_text = 'text' in kwargs and kwargs['text']
            os = int(kwargs.get('os', default_gzip_os))
            if os > 255 or os < 0: raise ValueError('Gzip OS is an invalid value')
            filename = get_filename(output)
            filename = filename[:-3] if filename and filename.endswith('.gz') else ''
            filename = gzip_header_str(kwargs.get('filename', filename))
            comment  = gzip_header_str(kwargs.get('comment',  ''))
            mtime    = int(kwargs.get('mtime', time()))
            extras = kwargs.get('extras')
            if extras and any(len(id) != 2 for id, data in extras): raise ValueError('Gzip extras had a subfield id that was not 2 characters long')
            self.gzip_options = {
                    'os' : os, 'mtime' : mtime, 'text' : is_text,
                    'filename' : filename, 'comment' : comment, 'extras' : extras
                }

        # Setup properties
        if isinstance(output, basestring):
            self.base = open(output, mode)
            self.owns_handle = True
        else:
            self.base = output
            self.owns_handle = False
        if type == None:
            self._base_buf = self.base.read(3)
            type = guesstype(self._base_buf)
        elif mode[0] == 'r':
            self._base_buf = ''
        self.type = type
        self.mode = mode
        self._writing = mode[0] != 'r'
        self._calc_checksum = checksums[self.type]
        self.name = get_filename(self.base, '')

        if self._writing:
            self._init_writing(level)
        else:
            self._init_reading()

    def _check(self, writing=None):
        """Raises a ValueError if the underlying file object has been closed."""
        if self.closed: raise ValueError('I/O operation on closed file.')
        if writing != None and self._writing != writing:
            raise ValueError('Cannot write to read-only file' if writing else 'Cannot read or rewind a write-only file')

    # Close
    @property
    def closed(self): return self.base is None
    def close(self):
        """
        If writing, completely flush the compressor and output the checksum if the format has it.
        Always close the file. If called more than once, subsequent calls are no-op.
        """
        if self.closed: return
        if self._writing:
            self.base.write(self.compressor.flush(Z_FINISH))
            del self.compressor
            if self.type == 'gzip':   self.base.write(pack('<LL', self.checksum, self.size & 0xffffffffL))
            elif self.type == 'zlib': self.base.write(pack('>L', self.checksum))
            del self._calc_checksum
            self.base.flush()
        if self.owns_handle: self.base.close()
        self.base = None

    # Random Properties
    def fileno(self):
        """Returns the file descriptor of the underlying file object."""
        return self.base.fileno()
    def isatty(self):
        """Returns True if the underlying file object is interactive."""
        return self.base.isatty()
    def __repr__(self):
        return '<gzip ' + repr(self.base) + ' at ' + hex(id(self)) + '>'

    # Position
    def seekable(self): return True
    def tell(self):
        self._check()
        return self.offset
    def rewind(self):
        """Return the uncompressed stream file position indicator to the beginning of the file"""
        self._check(False)
        self.base.seek(0)
        self._new_member = True
        self.extrabuf = ""
        self.extrasize = 0
        self.extrastart = 0
        self.offset = 0
    def seek(self, offset, whence=None):
        self._check()
        if whence:
            if whence == 1: offset += self.offset
            else: raise ValueError('Seek from end not supported') # whence == 2
        if self._writing:
            if offset < self.offset: raise IOError('Negative seek in write mode')
            count = offset - self.offset
            if count > 1024: zeros = 1024 * b'\0'
            for i in xrange(count // 1024): self.write(zeros)
            self.write((count % 1024) * b'\0')
        else:
            if offset < self.offset: self.rewind() # for negative seek, rewind and do positive seek
            count = offset - self.offset
            for i in xrange(count // 1024): self.read(1024)
            self.read(count % 1024)
        return self.offset

    # Writing
    def _init_writing(self, level):
        self.checksum = self._calc_checksum("") & 0xffffffffL
        self.size = 0
        windowsize = MAX_WBITS
        if self.type == 'gzip':
            flags = FHCRC
            if self.gzip_options['text']:     flags |= FTEXT
            if self.gzip_options['extras']:   flags |= FEXTRA
            if self.gzip_options['filename']: flags |= FNAME
            if self.gzip_options['comment']:  flags |= FCOMMENT
            xf = 2 if level >= 7 else (4 if level <= 2 else 0)
            s = b'\x1F\x8B\x08' + pack('<BLBB', flags, self.gzip_options['mtime'], xf, self.gzip_options['os'])
            self.base.write(s)
            chk16 = crc32(s) & 0xffffffffL
            if self.gzip_options['extras']:
                extras = ''
                for id, data in self.gzip_options['extras']:
                    extras += id + pack('<H', len(data)) + data
                extras = pack('<H', len(extras)) + extras
                chk16 = write_gzip_header_str(self.base, extras, chk16)
            if self.gzip_options['filename']: chk16 = write_gzip_header_str(self.base, self.gzip_options['filename'], chk16)
            if self.gzip_options['comment']:  chk16 = write_gzip_header_str(self.base, self.gzip_options['comment'],  chk16)
            self.base.write(pack('<H', chk16 & 0xffff))
        elif self.type == 'zlib':
            header = 0x7800 + (((level+1)//3) << 6)
            # Make header a multiple of 31
            mod31 = header % 31
            if mod31 != 0: header += (31 - mod31)
            self.base.write(pack('>H', header))
            windowsize = 15
        self.base.flush()
        self.compressor = compressobj(level, DEFLATED, -windowsize)
    def writable(self): return self._writing
    def write(self, data):
        """Compress the data and write to the underlying file object. Update the checksum."""
        self._check(True)
        if isinstance(data, memoryview): data = data.tobytes() # Convert data type if called by io.BufferedWriter
        if len(data) > 0:
            self.size += len(data)
            self.checksum = self._calc_checksum(data, self.checksum) & 0xffffffffL
            self.base.write(self.compressor.compress(data))
            self.offset += len(data)
        return len(data)
    def flush(self, full=False):
        """
        Flush the data from the compression buffer into the underlying file object. This will
        slightly decrease compression efficency. If full is True a more major flush is performed
        that will degrade compression more but does mean if the data is corrupted some
        decompression will be able to restart.
        """
        self._check()
        if self._writing:
            self.base.write(self.compressor.flush(Z_FULL_FLUSH if full else Z_SYNC_FLUSH))
            self.base.flush()

    # Reading
    def _init_reading(self):
        self._read_header()
        
        # Buffer data read from gzip file. extrastart is offset in
        # stream where buffer starts. extrasize is number of
        # bytes remaining in buffer from current stream position.
        self.extrabuf = ""
        self.extrasize = 0
        self.extrastart = 0
        
        # Starts small, scales exponentially
        self.min_readsize = 100
        
    def _read_base(self, n, check_eof = True):
        if n < len(self._base_buf):
            s = self._base_buf[:n]
            self._base_buf = self._base_buf[n:]
        elif len(self._base_buf) > 0:
            s = self._base_buf
            self._base_buf = ''
            s += self.base.read(n - len(s))
        else:
            s = self.base.read(n)
        if check_eof and len(s) != n: raise EOFError
        return s
    def _read_more(self, n, str):
        if len(str) > n:
            self._base_buf = str[n:]
            str = str[:n]
            return str
        elif len(str) == n: return str
        return str + self._read_base(n - len(str))
    def _read_header(self):
        self.checksum = self._calc_checksum("") & 0xffffffffL
        self.size = 0
        windowsize = MAX_WBITS
        if self.type == 'gzip':
            if not hasattr(self, 'gzip_options'):
                self.gzip_options = {
                        'os' : 255, 'mtime' : 0, 'text' : False, 
                        'filename' : None, 'comment' : None, 'extras' : None
                    }
            header = self._read_base(10)
            magic1, magic2, method, flags, mtime, xf, os = unpack('<BBBBIBB', header)
            if magic1 != 0x1F or magic2 != 0x8B: raise IOError('Not a gzipped file')
            if method != 8: raise IOError('Unknown compression method')
            if flags & 0xE0: raise IOError('Unknown flags')
            self.gzip_options['text'] = bool(flags & FTEXT)
            self.gzip_options['os'] = os
            self.gzip_options['mtime'] = mtime
            chk16 = crc32(header) & 0xffffffffL
            if flags & FEXTRA:
                # Read the extra field
                xlen = self._read_base(2)
                extras = self._read_base(unpack('<H', xlen)[0])
                chk16 = crc32(extras, crc32(xlen, chk16)) & 0xffffffffL
                ext = []
                while len(extras) >= 4:
                    l = unpack('<H', extras[2:4])[0]
                    if 4+l > len(extras): raise IOError('Invalid extra fields in header')
                    ext.append((extras[:2], extras[4:4+l]))
                    extras = extras[4+l:]
                if len(extras) > 0: raise IOError('Invalid extra fields in header')
                self.gzip_options['extras'] = ext
            if flags & FNAME:    self.gzip_options['filename'], chk16 = read_gzip_header_str(self._read_base, chk16)
            if flags & FCOMMENT: self.gzip_options['comment'],  chk16 = read_gzip_header_str(self._read_base, chk16)
            # Read and verify the 16-bit header CRC
            chk16_ = unpack('<H', self._read_base(2))[0]
            if (flags & FHCRC) and chk16_ != (chk16 & 0xffff): raise IOError('Header corrupted')
        elif self.type == 'zlib':
            header = self._read_base(2)
            header = unpack('>H', header)[0]
            method = (header >>  8) & 0xF
            windowsize = (header >> 12) & 0xF
            fdict  = (header & 0x20) != 0
            #flevel = (header >>  6) & 0x3
            #fcheck = (header & 0x1F)
            if method != 8 or window >= 8 or fdict: raise IOError('Unknown compression method')
            if header % 31 != 0: raise IOError('Header corrupted')
        self.decompressor = decompressobj(-windowsize)
        self._new_member = False
    def _read_footer(self, footer = ''):
        try:
            if self.type == 'gzip':
                footer = self._read_more(8, footer)
                crc32, isize = unpack('<II', footer)
                if crc32 != self.checksum:
                    raise IOError("CRC32 check failed %08x != %08x" % (crc32, self.checksum))
                elif isize != (self.size & 0xffffffffL):
                    raise IOError("Incorrect length of data produced")
            elif self.type == 'zlib':
                footer = self._read_more(4, footer)
                a32 = unpack('>I', footer)[0]
                if a32 != self.checksum:
                    raise IOError("Adler32 check failed %08x != %08x" % (a32, self.checksum))
            elif len(footer) > 0: self._read_more(0, footer)
        except EOFError: raise IOError("Corrupt file: did not end with checksums")
        # Skip any zero-padding
        c = "\x00"
        while c == "\x00": c = self._read_base(1, False)
        if c: self._base_buf = c + self._base_buf
        self._new_member = True

    def _read(self, size=1024):
        if self._new_member: self._read_header()
        buf = self._read_base(size, False)
        if len(buf) == 0:
            self._add_read_data(self.decompressor.flush())
            self._read_footer()
            raise EOFError
        self._add_read_data(self.decompressor.decompress(buf))
        if len(self.decompressor.unused_data) != 0:
            self._read_footer(self.decompressor.unused_data)
    def _add_read_data(self, data):
        self.checksum = self._calc_checksum(data, self.checksum) & 0xffffffffL
        offset = self.offset - self.extrastart
        self.extrabuf = self.extrabuf[offset:] + data
        self.extrasize += len(data)
        self.extrastart = self.offset
        self.size += len(data)
    def _unread(self, buf):
        self.extrasize += len(buf)
        self.offset -= len(buf)

    def readable(self): return not self._writing
    def read(self, size=-1):
        self._check(False)
        readsize = 1024
        if size < 0:        # get the whole thing
            try:
                while True:
                    self._read(readsize)
                    readsize = min(self.max_read_chunk, readsize * 2)
            except EOFError:
                size = self.extrasize
        else:               # just get some more of it
            try:
                while size > self.extrasize:
                    self._read(readsize)
                    readsize = min(self.max_read_chunk, readsize * 2)
            except EOFError:
                if size > self.extrasize:
                    size = self.extrasize
        offset = self.offset - self.extrastart
        chunk = self.extrabuf[offset:offset+size]
        self.extrasize -= size
        self.offset += size
        return chunk
    def readline(self, size=-1):
        if size < 0:
            # Shortcut common case - newline found in buffer
            offset = self.offset - self.extrastart
            i = self.extrabuf.find('\n', offset) + 1
            if i > 0:
                self.extrasize -= i - offset
                self.offset += i - offset
                return self.extrabuf[offset:i]
            size = maxint
            readsize = self.min_readsize
        else:
            readsize = size
        bufs = []
        while size != 0:
            c = self.read(readsize)
            i = c.find('\n') + 1

            # We set i=size to break out of the loop under two
            # conditions: 1) there's no newline, and the chunk is
            # larger than size, or 2) there is a newline, but the
            # resulting line would be longer than 'size'.
            if (size < i) or (i == 0 and len(c) > size):
                i = size

            if i > 0 or c == '':
                bufs.append(c[:i])    # Add portion of last chunk
                self._unread(c[i:])   # Push back rest of chunk
                break

            # Append chunk to list, decrease 'size',
            bufs.append(c)
            size -= len(c)
            readsize = min(size, readsize * 2)
        if readsize > self.min_readsize:
            self.min_readsize = min(readsize, self.min_readsize * 2, 512)
        return ''.join(bufs) # Return resulting line

if __name__ == '__main__':
##    import sys
##    
##    # Act like gzip; with -d, act like gunzip.
##    # The input file is not deleted, however, nor are any other gzip options or features supported.
##    args = sys.argv[1:]
##    func = compress_file
##    if args and (args[0] == '-d' or args[0] == '--decompress'):
##        args = args[1:]
##        func = decompress_file
##    if not args: args = ["-"]
##    for arg in args:
##        if arg == "-": func(sys.stdin, sys.stdout)
##        else:          func(arg)
    gzdata = (b'\x1f\x8b\x08\x04\xb2\x17cQ\x02\xff'
                   b'\x09\x00XX\x05\x00Extra'
                   b'\x0bI-.\x01\x002\xd1Mx\x04\x00\x00\x00')
