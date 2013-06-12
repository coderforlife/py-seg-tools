#!/usr/bin/env python

from utils import check_reqs
check_reqs(SimpleITK = False)

"""
Converts a PNG stack to an MRC file. Runs either as a command line program or as
an importable function.
"""

def png2mrc(pngs, mrc, flip = False, sigma = 0.0):
    """
    Converts a PNG stack to an MRC file. Returns the new MRC file object.

    Arguments:
    pngs     -- the PNGs to read, an iterable of file names
    mrc      -- the MRC to save to
    
    Optional Arguments:
    flip     -- if True then each image is flipped top to bottom before saving
    sigma    -- the amount of blurring to perform on the slices while saving, as the sigma argument for a Gaussian blur, defaults to no blurring
    """
    from os.path import join
    from images import sp_read, MRC, flip_up_down, gauss_blur

    pngs = iter(pngs)
    flip = bool(flip)
    sigma = float(sigma)

    try:
        png = sp_read(pngs.next())
    except StopIteration: raise ValueError("Must provide at least one PNG")
    mrc = MRC(mrc, nx=png.shape[1], ny=png.shape[0], dtype=png.dtype)
    mrc.append(png)
    mrc.append_all((sp_read(png) for png in pngs)) # will skip the first one
    mrc.write_header()
    return mrc

def help_msg(err = 0, msg = None):
    from os.path import basename
    from sys import stderr, argv, exit
    from textwrap import fill, TextWrapper
    from utils import get_terminal_width
    w = max(get_terminal_width(), 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] input1.png [input2.png ...] output.mrc" % basename(argv[0]))
    print ""
    print tw.fill("You may also use a glob-like syntax for any of the input files, such as 'folder/*.png' or '[0-9][0-9][0-9].png'")
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -f  --flip      If given then each image is flipped top to bottom before saving")
    print tw.fill("  -s  --sigma=    Sigma for Gaussian blurring while saving, defaults to no blurring")
    exit(err)
        
if __name__ == "__main__":
    from os.path import isfile, realpath
    from sys import argv
    from getopt import getopt, error as getopt_error
    from glob import iglob
    from images import MRC
    
    if len(argv) < 2: help_msg(1)
    
    try: opts, args = getopt(argv[1:], "hfs:", ["help", "flip", "sigma="])
    except getopt_error, msg: help_msg(2, msg)

    # Parse arguments
    flip = False
    sigma = None
    for o,a in opts:
        if o == "-h" or o == "--help":
            help_msg()
        elif o == "-f" or o == "--flip":
            if flip: help_msg(2, "Must be only one flip argument")
            flip = True
        elif o == "-s" or o == "--sigma":
            if sigma != None: help_msg(2, "Must be only one sigma argument")
            try: sigma = float(a)
            except: help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
            if sigma < 0 or isnan(sigma): help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")

    # Make sure paths are good
    if len(args) < 2: help_msg(2, "You need to provide at least one PNG path/glob and an MRC as arguments")
    mrc_filename = realpath(args[-1])
    pngs = []
    for png in args[:-1]:
        png = realpath(png)
        if not isfile(png):
            if '*' in png or '?' in png or ('[' in png and ']' in png):
                pngs.extend(iglob(png))
            else:
                help_msg(2, "PNG file does not exist: %s", png)
        else:
            pngs.append(png)
    if len(pngs) == 0: help_msg(2, "No PNGs were found using the given arguments")

    # Get default values for optional args
    if sigma == None: sigma = 0.0

    # Do the actual work!
    png2mrc(pngs, mrc_filename, flip, sigma)
