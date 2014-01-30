#!/usr/bin/env python

from utils import check_reqs
check_reqs()

"""
Converts an image stack to an MRC file. Runs either as a command line program or as an importable function.
"""

def stack2mrc(stack, mrc, imfilter = lambda im: im):
    """
    Converts an image stack to an MRC file. Returns the new MRC file object.

    Arguments:
    stack    -- the images to read, an iterable of file names
    mrc      -- the MRC filename to save to
    
    Optional Arguments:
    imfilter  -- a function/callable that takes and returns an image
    """
    from mrc import MRC
    from images import imread

    stack = iter(stack)
    try: img = imfilter(imread(stack.next()))
    except StopIteration: raise ValueError("Must provide at least one image")
    mrc = MRC(mrc, nx=img.shape[1], ny=img.shape[0], dtype=img.dtype)
    mrc.append(img)
    mrc.append_all(imfilter(imread(img)) for img in stack) # will skip the first one
    mrc.write_header()
    return mrc

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
    print tw.fill("  %s [args] input1.xxx [input2.xxx ...] output.mrc" % basename(argv[0]))
    print ""
    print "You may also use a glob-like syntax for any of the input files, such as 'folder/*.png' or '[0-9][0-9][0-9].png'"
    print ""
    print "Supports numerous file formats. MHA/MHD files must have the proper file extension. Other files will have their data examined to determine type. All images must have the same dimensions and pixel format. Not all pixel formats all supported."
    print ""
    print "Optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    for l in imfilter_util.usage: print tw.fill(l) if len(l) > 20 and l[0] == ' ' else l
    exit(err)
        
if __name__ == "__main__":
    from os.path import isfile, realpath
    from sys import argv
    from getopt import getopt, GetoptError
    from glob import iglob
    import imfilter_util
    from mrc import MRC
    
    if len(argv) < 2: help_msg(1)
    
    try: opts, args = getopt(argv[1:], "h"+imfilter_util.getopt_short, ["help"]+imfilter_util.getopt_long)
    except GetoptError as err: help_msg(2, str(err))

    # Parse arguments
    flip = False
    sigma = None
    for o,a in opts:
        if o == "-h" or o == "--help": help_msg()
        else: imfilters += [imfilter_util.parse_opt(o,a,help_msg)]

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
    imf = imfilter_util.list2imfilter(imfilters)

    # Do the actual work!
    stack2mrc(stack, mrc_filename, imf)
