#!/usr/bin/env python

"""
Converts an MRC file to an MHA stack. Runs either as a command line program or
as an importable function.
"""

def mrc2mha(mrc, mha_dir, indxs = None, basename = "%03d.mha", mode = None, sigma = 0.0):
    """
    Converts an MRC file to an MHA stack

    Arguments:
    mrc      -- the MRC file (as MRC object) or filepath (as a string)
    mha_dir  -- the directory to save MHAs to
    
    Optional Arguments:
    indxs    -- the indices of slices to save, default is to use all slices
    basename -- the template name to use for MHAs, needs to have a %d to be replaced by slice number, default is "%d.mha"
    mode     -- output mode, one of:
                    'float' to output a 32-bit floating-point number output scaled to 0.0-1.0
                    'label' to output a consecutively numbered image for label data
                    None (default) to perform no conversion
    sigma    -- the amount of blurring to perform on the slices while saving, as the sigma argument for a Gaussian blur, defaults to no blurring
    """
    from os.path import join
    from images import MRC, gauss_blur, float_image, create_labels, itk_save
    from utils import make_dir

    float_it = False
    label_it = False
    if mode == 'float': float_it = True
    elif mode == 'label': label_it = True
    elif mode != None: raise ValueError("Mode must be 'float', 'label', or None")
    if isinstance(mrc, basestring): mrc = MRC(mrc)
    if not make_dir(mha_dir): raise IOError("Unable to create directory")
    if indxs == None:
        for i, sec in enumerate(mrc):
            if sigma != 0.0: sec = gauss_blur(sec, sigma)
            if float_it: sec = float_image(sec)
            elif label_it: sec = create_labels(sec)
            itk_save(join(mha_dir, basename % i), sec)
    else:
        for i in indxs:
            sec = mrc[i]
            if sigma != 0.0: sec = gauss_blur(sec, sigma)
            if float_it: sec = float_image(sec)
            elif label_it: sec = create_labels(sec)
            itk_save(join(mha_dir, basename % i), sec)

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
    print tw.fill("  -b  --base      The base filename base to use, needs to have a %d to replace with slice number, defaults to '%03d.mha'")
    print tw.fill("  -i  --indices   The slice indices to use, accepts numbers with commas and dashes between them, default is all slices")
    print tw.fill("  -m  --mode      The output mode, either 'float' for scaled floating-point ouput or 'label' for consecutively numbered label data, default is neither")
    print tw.fill("  -s  --sigma     Sigma for Gaussian blurring while saving, defaults to no blurring")
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath
    from sys import argv
    from getopt import getopt, error as getopt_error
    from math import isnan
    
    from utils import make_dir, check_reqs
    check_reqs()

    from images import MRC

    if len(argv) < 2: help_msg(1)

    try:
        opts, args = getopt(argv[1:], "hb:i:m:s:", ["help", "base=", "indices=", "mode=", "sigma="])
    except getopt_error, msg: help_msg(2, msg)

    # Parse arguments
    indxs = None
    basename = None
    mode = None
    sigma = None
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
        elif o == "-i" or o == "--indices":
            if indxs != None: help_msg(2, "Must be only one indxs argument")
            parts = []
            for p in indxs.split(","):
                if p.isdigit(): # single digit
                    p = int(p)
                    if p < 0: help_msg(2, "Invalid indices argument supplied")
                    parts.append(p)
                else: # range of numbers
                    p = [int(p) for p in p.split('-') if p.isdigit()]
                    if len(p) != 2 or p[0] < 0 or p[1] < p[0]: help_msg(2, "Invalid indices argument supplied")
                    parts.extend(range(p[0], p[1] + 1))
            parts = list(set(parts)) # remove duplicates
            parts.sort()
        elif o == "-m" or o == "--mode":
            if mode != None: help_msg(2, "Must be only one mode argument")
            mode = a
            if mode != 'float' and mode != 'label': help_msg(2, "Mode must be either 'float' or 'label'")
        elif o == "-s" or o == "--sigma":
            if sigma != None: help_msg(2, "Must be only one sigma argument")
            try: sigma = float(a)
            except: help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
            if sigma < 0 or isnan(sigma): help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")

    # Make sure path are good
    if len(args) != 2: help_msg(2, "You need to provide an MRC and MHA output directory as arguments")
    mrc_filename = realpath(args[0])
    try: mrc = MRC(mrc_filename)
    except BaseException as e: help_msg(2, "Failed to open MRC file: " + str(e))
    mha_dir = realpath(args[1])
    if not make_dir(mha_dir): help_msg(2, "MHA output directory already exists as regular file, choose another directory")

    # Set defaults for optional args
    if sigma    == None: sigma = 0.0
    if basename == None: basename = "%03d.mha"

    # Do the actual work!
    mrc2mha(mrc, mha_dir, indxs, basename, mode, sigma)
