#!/usr/bin/env python

from utils import check_reqs
check_reqs()

"""
Converts an MRC file to an image stack. Runs either as a command line program or as an importable function.
"""

def mrc2stack(mrc, out_dir, indxs = None, basename = "%04d.png", imfilter = lambda im: im):
    """
    Converts an MRC file to an image stack

    Arguments:
    mrc      -- the MRC file (as MRC object) or filepath (as a string)
    out_dir  -- the directory to save the stack to
    
    Optional Arguments:
    indxs     -- the indices of slices to save, default is to use all slices
    basename  -- the template name to use for images, needs to have a %d to be replaced by slice number, default is "%04d.png"
    imfilter  -- a function/callable that takes and returns an image
    """
    from os.path import join
    from mrc import MRC
    from images import imsave
    from utils import make_dir

    if isinstance(mrc, basestring): mrc = MRC(mrc)
    if not make_dir(out_dir): raise IOError("Unable to create output directory")
    if indxs == None:
        for i, sec in enumerate(mrc): imsave(join(out_dir, basename % i), imfilter(sec))
    else:
        for i in indxs: imsave(join(out_dir, basename % i), imfilter(mrc[i]))

def help_msg(err = 0, msg = None):
    from os.path import basename
    from sys import stderr, argv, exit
    from textwrap import fill, TextWrapper
    from utils import get_terminal_width
    import imfilter_util
    w = max(get_terminal_width(), 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] input.mrc output_directory" % basename(argv[0]))
    print ""
    print tw.fill("Supports numerous file formats based on extension. Not all types can be saved to with all options.")
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -e  --ext=      The extension (type) of the files to save, defaults to 'png'")
    print tw.fill("  -b  --base=     The base filename (without extension) to use, needs to have a %d to replace with slice number, defaults to '%04d'")
    print tw.fill("  -x #-#          The x coordinate to extract given as two integers seperated by a dash")
    print tw.fill("  -y #-#          The y coordinate to extract given as two integers seperated by a dash")
    print tw.fill("  -z indices      The slice indices to use, accepts integers with commas and dashes between them")
    for l in imfilter_util.usage: print tw.fill(l)
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath
    from sys import argv
    from getopt import getopt, GetoptError
    import imfilter_util

    from utils import make_dir
    from mrc import MRC
    
    
    if len(argv) < 2: help_msg(1)
    
    try: opts, args = getopt(argv[1:], "he:b:x:y:z:"+imfilter_util.getopt_short, ["help", "ext=", "base="]+imfilter_util.getopt_long)
    except GetoptError as err: help_msg(2, str(err))

    # Parse arguments
    x = None
    y = None
    z = None
    ext = None
    basename = None
    imfilters = []
    for o,a in opts:
        if o == "-h" or o == "--help": help_msg()
        elif o == "-e" or o == "--ext":
            if basename != None: help_msg(2, "Must be only one extension argument")
            ext = a
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
            x = x.split("-")
            if len(x) != 2 or not x[0].isdigit() or not x[1].isdigit(): help_msg(2, "Invalid x argument supplied")
            x = (int(x[0]), int(x[1]))
        elif o == "-y":
            if y != None: help_msg(2, "May be only one y argument")
            y = y.split("-")
            if len(y) != 2 or not y[0].isdigit() or not y[1].isdigit(): help_msg(2, "Invalid y argument supplied")
            y = (int(y[0]), int(y[1]))
        else: imfilters += [imfilter_util.parse_opt(o,a,help_msg)]

    # Make sure paths are good
    if len(args) != 2: help_msg(2, "You need to provide an MRC and image output directory as arguments")
    mrc_filename = realpath(args[0])
    try: mrc = MRC(mrc_filename, readonly=True)
    except BaseException as e: help_msg(2, "Failed to open MRC file: " + str(e))
    out_dir = realpath(args[1])
    if not make_dir(out_dir): help_msg(2, "Output directory already exists as regular file, choose another directory")

    # Check other arguments, getting values for optional args, etc.
    ext = "png" if ext == None else ext.lstrip('.')
    if basename == None: basename = "%04d"
    if x == None: x = (0, mrc.nx - 1)
    elif x[0] < 0 or x[1] < x[0] or x[1] >= mrc.nx: help_msg(2, "Invalid x argument supplied")
    if y == None: y = (0, mrc.ny - 1)
    elif y[0] < 0 or y[1] < y[0] or y[1] >= mrc.ny: help_msg(2, "Invalid x argument supplied")
    if z:
        min_z, max_z = min(z), max(z)
        if min_z < 0 or max_z >= mrc.nz: help_msg(2, "Invalid z argument supplied")
    imf = imfilter_util.list2imfilter(imfilters)

    # Do the actual work!
    mrc2stack(mrc.view(x, y, (0, mrc.nz - 1)), out_dir, z, basename+"."+ext, imf)
