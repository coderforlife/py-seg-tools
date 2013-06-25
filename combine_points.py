#!/usr/bin/env python

"""
Combines many pts files (for use with points2model) into a single file, renumbering the contours.
Currently only accepts the format with 4 entries per line: contour #, x, y, z.
"""

def combine_points(input_files, output_file):
    """
    Combines many pts files (for use with points2model) into a single file, renumbering the contours.
    Currently only accepts the format with 4 entries per line: contour #, x, y, z.
    The input files need to be already sorted by contour number.
    input_files is an iterable of file names, in the order you want them to be combined.
    output_files is the single output file to save results to
    """
    #from  import 
    with open(output_file, "wb") as out:
        out_cn = 0 # actual starting value is 1
        last_cn = None
        for i in input_files:
            with open(i, "r") as f:
                for line in f:
                    vals = line.split()
                    if len(vals) == 0: continue
                    if len(vals) != 4: raise ValueError("File "+i+" not in the right format")
                    try: vals = [int(x) for x in vals]
                    except ValueError: raise ValueError("File "+i+" not in the right format")
                    if vals[0] != last_cn:
                        last_cn = vals[0]
                        out_cn += 1

                    vals[0] = out_cn
                    out.write(" ".join((str(x) for x in vals)))
                    out.write("\n")

def help_msg(err = 0, msg = None):
    from os.path import basename
    from sys import stderr, argv, exit
    from textwrap import fill, TextWrapper
    from utils import get_terminal_width
    w = max(get_terminal_width(), 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s input.pts [input2.pts ...] output.pts" % basename(argv[0]))
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath, exists
    from sys import argv
    from getopt import getopt, error as getopt_error

    if len(argv) < 2: help_msg(1)

    try:
        opts, args = getopt(argv[1:], "h", ["help"])
    except getopt_error, msg: help_msg(2, msg)

    # Parse arguments
    for o,a in opts:
        if o == "-h" or o == "--help":
            help_msg()

    # Make sure path are good
    if len(args) < 2: help_msg(2, "You need to provide at least one input points file and an output points file as arguments")
    inputs = [realpath(i) for i in args[:-1] if exists(i)]
    if len(inputs) != len(args) - 1: help_msg(2, "At least one of the input points file does not exist")
    output = realpath(args[-1])
    if sigma == None: sigma = 0.0

    # Do the actual work!
    combine_points(inputs, output)

