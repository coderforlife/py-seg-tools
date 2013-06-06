#!/usr/bin/env python

from utils import check_reqs
check_reqs()

"""
Converts an MRC file to a PNG stack. Runs either as a command line program or as
an importable function.
"""

def mrc2png(mrc, png_dir, indxs = None, basename = "%03d.png"):
    """
    Converts an MRC file to a PNG stack

    Arguments:
    mrc      -- the MRC file (as MRC object) or filepath (as a string)
    png_dir  -- the directory to save PNGs to
    
    Optional Arguments:
    indxs    -- the indices of slices to save, default is to use all slices
    basename -- the template name to use for PNGs, needs to have a %d to be replaced by slice number, default is "%03d.png"  
    """
    from os.path import join
    from images import MRC, sp_save
    from utils import make_dir

    if isinstance(mrc, basestring): mrc = MRC(mrc)
    if not make_dir(png_dir): raise IOError("Unable to create directory")
    if indxs == None:
        for i, sec in enumerate(mrc):
            sp_save(join(png_dir, basename % i), sec)
    else:
        for i in indxs:
            sp_save(join(png_dir, basename % i), mrc[i])

def help_msg(err = 0, msg = None):
    from os.path import basename
    from sys import stderr, argv, exit
    from textwrap import fill, TextWrapper
    from utils import get_terminal_width
    w = max(get_terminal_width(), 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] input.mrc output_directory" % basename(argv[0]))
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -b  --base      The base filename base to use, needs to have a %d to replace with slice number, defaults to '%03d.png'")
    print tw.fill("  -x              The x coordinate to extract given as two integers seperated by a comma")
    print tw.fill("  -y              The y coordinate to extract given as two integers seperated by a comma")
    print tw.fill("  -z              The slice indices to use, accepts integers with commas and dashes between them")
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath
    from sys import argv
    from getopt import getopt, error as getopt_error

    from utils import make_dir, check_reqs
    check_reqs()

    from images import MRC
    
    
    if len(argv) < 2: help_msg(1)
    
    try: opts, args = getopt(argv[1:], "hb:x:y:z:", ["help", "base="])
    except getopt_error, msg: help_msg(2, msg)

    # Parse arguments
    x = None
    y = None
    z = None
    basename = None
    for o,a in opts:
        if o == "-h" or o == "--help":
            help_msg()
        elif o == "-b" or o == "--base":
            if basename != None: help_msg(2, "Must be only one basename argument")
            try:
                _ = a % (1)
            except:
                help_msg(2, "The basename must contain %d (or a variant) for the slice number")
            basename = a
        elif o == "-z":
            if z != None: help_msg(2, "Must be only one z argument")
            z = []
            for p in a.split(","):
                if p.isdigit(): # single digit
                    p = int(p)
                    if p < 0: help_msg(2, "Invalid z argument supplied")
                    z.append(p)
                else: # range of numbers
                    p = [int(p) for p in p.split('-') if p.isdigit()]
                    if len(p) != 2 or p[0] < 0 or p[1] < p[0]: help_msg(2, "Invalid z argument supplied")
                    z.extend(range(p[0], p[1] + 1))
            z = list(set(z)) # remove duplicates
            z.sort()
        elif o == "-x":
            if x != None: help_msg(2, "May be only one x argument")
            x = x.split(",")
            if len(x) != 2 or not x[0].isdigit() or not x[1].isdigit(): help_msg(2, "Invalid x argument supplied")
            x = (int(x[0]), int(x[1]))
        elif o == "-y":
            if y != None: help_msg(2, "May be only one y argument")
            y = y.split(",")
            if len(y) != 2 or not y[0].isdigit() or not y[1].isdigit(): help_msg(2, "Invalid y argument supplied")
            y = (int(y[0]), int(y[1]))

    # Make sure paths are good
    if len(args) != 2: help_msg(2, "You need to provide an MRC and PNG output directory as arguments")
    mrc_filename = realpath(args[0])
    try: mrc = MRC(mrc_filename, readonly=True)
    except BaseException as e: help_msg(2, "Failed to open MRC file: " + str(e))
    png_dir = realpath(args[1])
    if not make_dir(png_dir): help_msg(2, "PNG output directory already exists as regular file, choose another directory")

    # Check other arguments, getting values for optional args, etc.
    if basename == None: basename = "%03d.png"
    if x == None: x = (0, mrc.nx - 1)
    elif x[0] < 0 or x[1] < x[0] or x[1] < mrc.nx: help_msg(2, "Invalid x argument supplied")
    if y == None: y = (0, mrc.ny - 1)
    elif y[0] < 0 or y[1] < y[0] or y[1] < mrc.ny: help_msg(2, "Invalid x argument supplied")
    zs = (min(z), max(z)) if z else (0, mrc.nz - 1)

    # Do the actual work!
    mrc2png(mrc.view(x, y, zs), png_dir, z, basename)
