#!/usr/bin/env python

from utils import check_reqs
check_reqs()

"""
Converts an image file to a new format.
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
    print tw.fill("  %s [args] input.xxx output.xxx" % basename(argv[0]))
    print ""
    print tw.fill("Supports numerous file formats based on extension. The extension should be accurate to the filetype otherwise it may not work.")
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    for l in imfilter_util.usage: print tw.fill(l)
    exit(err)
        
if __name__ == "__main__":
    from os.path import realpath, exists
    from sys import argv
    from getopt import getopt, GetoptError
    import imfilter_util
    from images import imread, imsave

    if len(argv) < 2: help_msg(1)

    try: opts, args = getopt(argv[1:], "h"+imfilter_util.getopt_short, ["help"]+imfilter_util.getopt_long)
    except GetoptError as err: help_msg(2, str(err))

    # Parse arguments
    imfilters = []
    for o,a in opts:
        if o == "-h" or o == "--help": help_msg()
        else: imfilters += [imfilter_util.parse_opt(o,a,help_msg)]

    # Make sure path are good
    if len(args) != 2: help_msg(2, "You need to provide an input and output file as arguments")
    input = realpath(args[0])
    if not exists(input): help_msg(2, "Input file does not exist")
    output = realpath(args[1])

    # Set defaults for optional args
    imf = imfilter_util.list2imfilter(imfilters)

    # Do the actual work!
    imsave(output, imf(imread(input)))
