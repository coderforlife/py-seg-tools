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
    print tw.fill("  -b  --base      The base filename base to use, needs to have a %d to replace with slice number, defaults to '%d.png'")
    print tw.fill("  -i  --indices   The slice indices to use, accepts numbers with commas and dashes between them, default is all slices")
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath
    from sys import argv
    from getopt import getopt, error as getopt_error

    from utils import make_dir, check_reqs
    check_reqs()

    from images import MRC
    
    
    if len(argv) < 2: help_msg(1)
    
    try:
        opts, args = getopt(argv[1:], "hb:i:", ["help", "base=", "indices="])
    except getopt_error, msg: help_msg(2, msg)

    # Parse arguments
    indxs = None
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

    # Make sure path are good
    if len(args) != 2: help_msg(2, "You need to provide an MRC and PNG output directory as arguments")
    mrc_filename = realpath(args[0])
    try: mrc = MRC(mrc_filename)
    except BaseException as e: help_msg(2, "Failed to open MRC file: " + str(e))
    png_dir = realpath(args[1])
    if not make_dir(png_dir): help_msg(2, "PNG output directory already exists as regular file, choose another directory")

    # Set defaults for optional args
    if basename == None: basename = "%03d.png"

    # Do the actual work!
    mrc2png(mrc, png_dir, indxs, basename)
