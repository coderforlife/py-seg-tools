#!/usr/bin/env python

from utils import check_reqs
check_reqs()

"""
Converts an MRC file to an image stack. Runs either as a command line program or
as an importable function.
"""

def mrc2stack(mrc, out_dir, indxs = None, basename = "%04d.png", mode = None, flip = False, sigma = 0.0):
    """
    Converts an MRC file to an image stack

    Arguments:
    mrc      -- the MRC file (as MRC object) or filepath (as a string)
    out_dir  -- the directory to save the stack to
    
    Optional Arguments:
    indxs    -- the indices of slices to save, default is to use all slices
    basename -- the template name to use for images, needs to have a %d to be replaced by slice number, default is "%04d.png"
    mode     -- output mode, one of:
                    'float' to output a 32-bit floating-point number output scaled to 0.0-1.0
                    'label' to output a consecutively numbered image using connected components
                    'relabel' to output a consecutively numbered image from an already labeled image
                    None (default) to perform no conversion
    flip     -- if True then each image is flipped top to bottom before saving
    sigma    -- the amount of blurring to perform on the slices while saving, as the sigma argument for a Gaussian blur, defaults to no blurring
    """
    from os.path import join
    from mrc import MRC
    from images import flip_up_down, gauss_blur, float_image, label, relabel, imsave
    from utils import make_dir

    float_it = False
    relabel_it = False
    label_it = False
    if mode == 'float': float_it = True
    elif mode == 'label': label_it = True
    elif mode == 'relabel': relabel_it = True
    elif mode != None: raise ValueError("Mode must be 'float', 'label', 'relabel', or None")
    if isinstance(mrc, basestring): mrc = MRC(mrc)
    if not make_dir(out_dir): raise IOError("Unable to create output directory")
    flip = bool(flip)
    sigma = float(sigma)
    if indxs == None:
        for i, sec in enumerate(mrc):
            if flip: sec = flip_up_down(sec)
            if sigma != 0.0: sec = gauss_blur(sec, sigma)
            if float_it: sec = float_image(sec)
            elif label_it: im = label(sec)
            elif relabel_it: im = relabel(sec)
            imsave(join(out_dir, basename % i), sec)
    else:
        for i in indxs:
            sec = mrc[i]
            if flip: sec = flip_up_down(sec)
            if sigma != 0.0: sec = gauss_blur(sec, sigma)
            if float_it: sec = float_image(sec)
            elif label_it: im = label(sec)
            elif relabel_it: im = relabel(sec)
            imsave(join(out_dir, basename % i), sec)

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
    print tw.fill("Supports numerous file formats based on extension. Not all types can be saved to with all options.")
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -e  --ext=      The extension (type) of the files to save, defaults to 'png'")
    print tw.fill("  -b  --base=     The base filename (without extension) to use, needs to have a %d to replace with slice number, defaults to '%04d'")
    print tw.fill("  -x #-#          The x coordinate to extract given as two integers seperated by a dash")
    print tw.fill("  -y #-#          The y coordinate to extract given as two integers seperated by a dash")
    print tw.fill("  -z indices      The slice indices to use, accepts integers with commas and dashes between them")
    print tw.fill("  -f  --flip      If given then each image is flipped top to bottom before saving")
    print tw.fill("  -m  --mode=     The output mode, either 'float' for scaled floating-point ouput, 'label' for consecutively numbered label data using connected components, or 'relabel' for renumbering an image, default is none")
    print tw.fill("  -s  --sigma=    Sigma for Gaussian blurring while saving, defaults to no blurring")
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath
    from sys import argv
    from getopt import getopt, error as getopt_error
    from math import isnan

    from utils import make_dir
    from mrc import MRC
    
    
    if len(argv) < 2: help_msg(1)
    
    try: opts, args = getopt(argv[1:], "hfe:b:x:y:z:m:s:", ["help", "flip", "ext=", "base=", "mode=", "sigma="])
    except getopt_error, msg: help_msg(2, msg)

    # Parse arguments
    flip = False
    x = None
    y = None
    z = None
    ext = None
    basename = None
    mode = None
    sigma = None
    for o,a in opts:
        if o == "-h" or o == "--help":
            help_msg()
        elif o == "-f" or o == "--flip":
            if flip: help_msg(2, "Must be only one flip argument")
            flip = True
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
        elif o == "-m" or o == "--mode":
            if mode != None: help_msg(2, "Must be only one mode argument")
            mode = a
            if mode != 'float' and mode != 'label' and mode != 'relabel': help_msg(2, "Mode must be either 'float', 'label', or 'relabel'")
        elif o == "-s" or o == "--sigma":
            if sigma != None: help_msg(2, "Must be only one sigma argument")
            try: sigma = float(a)
            except: help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
            if sigma < 0 or isnan(sigma): help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")

    # Make sure paths are good
    if len(args) != 2: help_msg(2, "You need to provide an MRC and image output directory as arguments")
    mrc_filename = realpath(args[0])
    try: mrc = MRC(mrc_filename, readonly=True)
    except BaseException as e: help_msg(2, "Failed to open MRC file: " + str(e))
    out_dir = realpath(args[1])
    if not make_dir(out_dir): help_msg(2, "Output directory already exists as regular file, choose another directory")

    # Check other arguments, getting values for optional args, etc.
    if ext      == None: ext = "png"
    else:                ext = ext.lstrip('.')
    if basename == None: basename = "%04d"
    if sigma    == None: sigma = 0.0
    if x == None: x = (0, mrc.nx - 1)
    elif x[0] < 0 or x[1] < x[0] or x[1] >= mrc.nx: help_msg(2, "Invalid x argument supplied")
    if y == None: y = (0, mrc.ny - 1)
    elif y[0] < 0 or y[1] < y[0] or y[1] >= mrc.ny: help_msg(2, "Invalid x argument supplied")
    if z:
        min_z, max_z = min(z), max(z)
        if min_z < 0 or max_z >= mrc.nz: help_msg(2, "Invalid z argument supplied")

    # Do the actual work!
    mrc2stack(mrc.view(x, y, (0, mrc.nz - 1)), out_dir, z, basename+"."+ext, mode, flip, sigma)
