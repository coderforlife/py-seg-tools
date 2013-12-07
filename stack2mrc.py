#!/usr/bin/env python

from utils import check_reqs
check_reqs()

"""
Converts an image stack to an MRC file. Runs either as a command line program or
as an importable function.
"""

def stack2mrc(stack, mrc, flip = False, sigma = 0.0):
    """
    Converts an image stack to an MRC file. Returns the new MRC file object.

    Arguments:
    stack    -- the images to read, an iterable of file names
    mrc      -- the MRC filename to save to
    
    Optional Arguments:
    flip     -- if True then each image is flipped top to bottom before saving
    sigma    -- the amount of blurring to perform on the slices while saving, as the sigma argument for a Gaussian blur, defaults to no blurring
    """
    from os.path import join
    from mrc import MRC
    from images import imread, flip_up_down, gauss_blur

    stack = iter(stack)
    flip = bool(flip)
    sigma = float(sigma)

    if flip:
        read = (lambda f: gauss_blur(flip_up_down(imread(x)), sigma)) if sigma != 0.0 else (lambda f: flip_up_down(imread(x)))
    elif sigma != 0.0:
        read = lambda f: gauss_blur(imread(x), sigma)
    else:
        read = imread

    try:
        img = read(stack.next())
    except StopIteration: raise ValueError("Must provide at least one image")
    mrc = MRC(mrc, nx=img.shape[1], ny=img.shape[0], dtype=img.dtype)
    mrc.append(img)
    mrc.append_all((read(img) for img in stack)) # will skip the first one
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
    print tw.fill("Supports numerous file formats. MHA/MHD files must have the proper file extension. Other files will have their data examined to determine type. All images must have the same dimensions and pixel format. Not all pixel formats all supported.")
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
    from mrc import MRC
    
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
    if len(args) < 2: help_msg(2, "You need to provide at least one image path/glob and an MRC as arguments")
    mrc_filename = realpath(args[-1])
    stack = []
    for img in args[:-1]:
        img = realpath(img)
        if not isfile(img):
            if '*' in img or '?' in img or ('[' in img and ']' in img):
                stack.extend(sorted(iglob(img)))
            else:
                help_msg(2, "Image file does not exist: %s", img)
        else:
            stack.append(img)
    if len(stack) == 0: help_msg(2, "No images were found using the given arguments")

    # Get default values for optional args
    if sigma == None: sigma = 0.0

    # Do the actual work!
    stack2mrc(stack, mrc_filename, flip, sigma)
