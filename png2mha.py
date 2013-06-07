#!/usr/bin/env python

"""
Converts PNG file to an MHA file. Runs either as a command line program or as an
importable function.
"""

def png2mha(png, mha, mode = None, flip = False, sigma = 0.0):
    """
    Converts a PNG file to an MHA file

    Arguments:
    png      -- the input PNG filepath
    mha      -- the otuput MHA filepath
    
    Optional Arguments:
    mode     -- output mode, one of:
                    'float' to output a 32-bit floating-point number output scaled to 0.0-1.0
                    'label' to output a consecutively numbered image for label data
                    None (default) to perform no conversion
    flip     -- if True then image is flipped top to bottom before saving
    sigma    -- the amount of blurring to perform on the slices while saving, as the sigma argument for a Gaussian blur, defaults to no blurring
    """
    from images import sp_read, flip_up_down, gauss_blur, float_image, create_labels, itk_save

    float_it = False
    label_it = False
    if mode == 'float': float_it = True
    elif mode == 'label': label_it = True
    elif mode != None: raise ValueError("Mode must be 'float', 'label', or None")
    im = sp_read(png)
    if flip: sec = flip_up_down(sec)
    if sigma: im = gauss_blur(im, sigma)
    if float_it: im = float_image(im)
    elif label_it: im = create_labels(im)
    itk_save(mha, im)

def help_msg(err = 0, msg = None):
    from os.path import basename
    from sys import stderr, argv, exit
    from textwrap import fill, TextWrapper
    from utils import get_terminal_width
    w = max(get_terminal_width(), 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] input.png output.mha" % basename(argv[0]))
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -f  --flip      If given then image is flipped top to bottom before saving")
    print tw.fill("  -m  --mode=     The output mode, either 'float' for scaled floating-point ouput or 'label' for consecutively numbered label data, default is neither")
    print tw.fill("  -s  --sigma=    Sigma for Gaussian blurring while saving, defaults to no blurring")
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath, exists
    from sys import argv
    from getopt import getopt, error as getopt_error
    from math import isnan
    
    from utils import check_reqs
    check_reqs()

    if len(argv) < 2: help_msg(1)

    try:
        opts, args = getopt(argv[1:], "hfm:s:", ["help", "flip", "mode=", "sigma="])
    except getopt_error, msg: help_msg(2, msg)

    # Parse arguments
    flip = False
    mode = None
    sigma = None
    for o,a in opts:
        if o == "-h" or o == "--help":
            help_msg()
        elif o == "-f" or o == "--flip":
            if flip: help_msg(2, "Must be only one flip argument")
            flip = True
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
    if len(args) != 2: help_msg(2, "You need to provide a PNG and MHA file as arguments")
    png = realpath(args[0])
    if not exists(png): help_msg(2, "PNG file does not exist")
    mha = realpath(args[1])

    # Set defaults for optional args
    if sigma == None: sigma = 0.0

    # Do the actual work!
    png2mha(png, mha, mode, flip, sigma)
