def check_reqs(scipy = True, numpy = True, PIL = True, psutil = False):
    """Checks to see if the required 3rd-party modules are available for import."""
    def __die(name):
        from sys import stderr, exit
        print >> stderr, "Could not import the required module %s" % name
        print >> stderr, "Try running 'easy_install %s' to install it" % name
        exit(1)
    if scipy:
        try: import scipy
        except: __die('scipy')
    if numpy:
        try: import numpy
        except: __die('numpy')
    if PIL:
        try: import PIL
        except: __die('PIL')
    if psutil:
        try: import psutil
        except: __die('psutil')


def make_dir(d):
    """Makes a directory tree. If the path exists as a regular file already False is returned."""
    import os, os.path
    if os.path.isdir(d): return True
    if os.path.exists(d): return False
    try:
        os.makedirs(d)
        return True
    except: return False


def only_keep_num(d, allowed, match_slice = slice(None), pattern='*'):
    """
    Searches for all files matching a particular glob pattern, extracts the given slice as an
    integer, and makes sure it is in the list of allowed numbers. If not, the file is deleted.
    """
    from glob import iglob
    from os import unlink
    from os.path import basename, join, isfile
    
    files = ((f, basename(f)[match_slice]) for f in iglob(join(d, pattern)) if isfile(f))
    for f in (f for f, x in files if x.isdigit() and int(x) not in allowed):
        try: unlink(f)
        except Exception, e: pass
    files = ((f, basename(f)[match_slice]) for f in iglob(join(d, '.'+pattern)) if isfile(f))
    for f in (f for f, x in files if x.isdigit() and int(x) not in allowed):
        try: unlink(f)
        except Exception, e: pass


def get_terminal_width():
    """Gets the width of the terminal if there is a terminal, in which case 80 is returned."""
    from os import environ
    from sys import platform
    
    if platform == "win32":
        # Windows
        from ctypes import windll, c_short, c_ushort, c_int, c_uint, c_void_p, byref, Structure
        class COORD(Structure): _fields_ = [("X", c_short), ("Y", c_short)]
        class SMALL_RECT(Structure): _fields_ = [("Left", c_short), ("Top", c_short), ("Right", c_short), ("Bottom", c_short)]
        class CONSOLE_SCREEN_BUFFER_INFO(Structure): _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD), ("wAttributes", c_ushort), ("srWindow", SMALL_RECT), ("dwMaximumWindowSize", COORD)]
        GetStdHandle = windll.kernel32.GetStdHandle
        GetStdHandle.argtypes, GetStdHandle.restype = [c_uint], c_void_p
        GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
        GetConsoleScreenBufferInfo.argtypes, GetConsoleScreenBufferInfo.restype = [c_void_p, c_void_p], c_int
        def con_width(handle):
            handle = GetStdHandle(handle)
            if handle and handle != -1:
                csbi = CONSOLE_SCREEN_BUFFER_INFO()
                if GetConsoleScreenBufferInfo(handle, byref(csbi)): return csbi.dwSize.X
            return
        w = con_width(-11) or con_width(-12) or con_width(-10) # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE, STD_INPUT_HANDLE

    else:
        # *nix
        def ioctl_GWINSZ(fd):
            try:
                import fcntl, termios, struct
                return struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))[1]
            except: return
        w = ioctl_GWINSZ(1) or ioctl_GWINSZ(2) or ioctl_GWINSZ(0) # stdout, stderr, stdin
        if not w:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                w = ioctl_GWINSZ(fd)
                os.close(fd)
            except: pass

    # Last resort, mainly for *nix, but also set the default of 80
    if not w: w = environ.get('COLUMNS', 80)
    return int(w)

def iter_same(x): """Generator/iterator that always produces the given value""" yield x
