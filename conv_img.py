#!/usr/bin/env python

from utils import check_reqs
check_reqs()

"""
Converts an image file to a new format, possibly changing the image 'mode'. Runs
either as a command line program or as an importable function.
"""

def conv_img(input, output, mode = None, flip = False, sigma = 0.0):
    """
    Converts an image file.

    Arguments:
    input    -- the input filepath
    output   -- the otuput filepath
    
    Optional Arguments:
    mode     -- output mode, one of:
                    'float' to output a 32-bit floating-point number output scaled to 0.0-1.0
                    'label' to output a consecutively numbered image using connected components
                    'relabel' to output a consecutively numbered image from an already labeled image
                    None (default) to perform no conversion
    flip     -- if True then image is flipped top to bottom before saving
    sigma    -- the amount of blurring to perform on the slices while saving, as the sigma argument for a Gaussian blur, defaults to no blurring
    """
    from images import imread, flip_up_down, gauss_blur, float_image, label, relabel, imsave

    float_it = False
    relabel_it = False
    label_it = False
    if mode == 'float': float_it = True
    elif mode == 'label': label_it = True
    elif mode == 'relabel': relabel_it = True
    elif mode != None: raise ValueError("Mode must be 'float', 'label', 'relabel', or None")
    im = imread(input)
    if flip: sec = flip_up_down(sec)
    if sigma: im = gauss_blur(im, sigma)
    if float_it: im = float_image(im)
    elif label_it: im = label(im)
    elif relabel_it: im = relabel(im)
    imsave(output, im)

def help_msg(err = 0, msg = None):
    from os.path import basename
    from sys import stderr, argv, exit
    from textwrap import fill, TextWrapper
    from utils import get_terminal_width
    w = max(get_terminal_width(), 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] input.xxx output.xxx" % basename(argv[0]))
    print ""
    print tw.fill("Supports numerous file formats based on extension. The extension should be accurate to the filetype otherwise it may not work.")
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -f  --flip      If given then image is flipped top to bottom before saving")
    print tw.fill("  -m  --mode=     The output mode, either 'float' for scaled floating-point ouput, 'label' for consecutively numbered label data using connected components, or 'relabel' for renumbering an image, default is none")
    print tw.fill("  -s  --sigma=    Sigma for Gaussian blurring while saving, defaults to no blurring")
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath, exists
    from sys import argv
    from getopt import getopt, error as getopt_error
    from math import isnan

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
            if mode != 'float' and mode != 'label' and mode != 'relabel': help_msg(2, "Mode must be either 'float', 'label', or 'relabel'")
        elif o == "-s" or o == "--sigma":
            if sigma != None: help_msg(2, "Must be only one sigma argument")
            try: sigma = float(a)
            except: help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
            if sigma < 0 or isnan(sigma): help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")

    # Make sure path are good
    if len(args) != 2: help_msg(2, "You need to provide an input and output file as arguments")
    input = realpath(args[0])
    if not exists(input): help_msg(2, "Input file does not exist")
    output = realpath(args[1])

    # Set defaults for optional args
    if sigma == None: sigma = 0.0

    # Do the actual work!
    conv_img(input, output, mode, flip, sigma)
