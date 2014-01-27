#!/usr/bin/env python

from utils import check_reqs
check_reqs()

"""
Gets the histogram for image(s).
"""

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
    print tw.fill("  %s [args] input1.xxx [input2.xxx ...] output.txt" % basename(argv[0]))
    print ""
    print tw.fill("You may also use a glob-like syntax for any of the input files, such as 'folder/*.png' or '[0-9][0-9][0-9].png'. Outputs to stdout if output is -.")
    print ""
    print tw.fill("Supports numerous file formats. Sums the histogram from all given images. Saves a list of integers to the output file.")
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -n  --nbins=    Number of bins in the histogram, defaults to 256")
    for l in imfilter_util.usage: print tw.fill(l)
    exit(err)

if __name__ == "__main__":
    from os.path import isfile, realpath
    from sys import argv
    from getopt import getopt, error as getopt_error
    from glob import iglob
    from images import imhist
    from numpy import savetxt

    if len(argv) < 2: help_msg(1)

    try: opts, args = getopt(argv[1:], "h", ["help"])
    except getopt_error, msg: help_msg(2, msg)

    # Parse arguments
    nbins = None
    for o,a in opts:
        if o == "-h" or o == "--help": help_msg()
        elif o == "-n" or o == "--nbins":
            if nbins != None: help_msg(2, "Must be only one nbins argument")
            if not a.isdigit(): help_msg(2, "Number of bins must be an positive integer")
            nbins = int(a)
            if nbins < 1: help_msg(2, "Number of bins must be an positive integer")
        else: help_msg(2, "Invalid argument: "+o)
    

    # Make sure path are good
    if len(args) < 2: help_msg(2, "You need to provide at least one image path/glob and an output file as arguments")
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
    from sys import stdout
    out_filename = stdout if args[-1] == '-' else realpath(args[-1])
    
    # Get default values for optional args
    if nbins == None: nbins = 256

    # Do the actual work!
    savetxt(out_filename, imhist(stack, nbins), fmt='%d')
