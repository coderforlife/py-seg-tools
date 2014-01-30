__all__ = [ 'getopt_short', 'getopt_long', 'usage', 'parse_opt', 'list2imfilter' ]

# not -h, -e, -b, -x, -y, -z
getopt_short = "fFlrm:s:t:H:p:c:m:0:"
getopt_long = ["flip", "float", "label", "relabel", "sigma=", "thresh=", "histeq=", "pad=", "crop=", "mirror=", "bgzero="]
usage = (
    "  -f  --flip      If given then each image is flipped top to bottom before saving",
    "  -F  --float     Output a floating-point image from 0 to 1",
    "  -l  --label     Output a consecutively numbered label data using connected components",
    "  -r  --relabel   Output a renumbered label image (correcting for missing or split regions)",
    "  -s  --sigma=    Sigma for Gaussian blurring while saving, defaults to no blurring",
    "  -t  --thresh=   Convert image to black and white with the given threshold (values below are 0, values above and included are 1 - reversed with negative values)",
    "  -H  --histeq=   Perform histogram equalization; if value is a number then equal-sized bins are used, otherwise it needs to be a file of integers (or - for stdin)",
    "  -p  --pad=      Pad the image with 0s, given as 4 integers for the amount to add to the top, left, bottom, and right",
    "  -c  --crop=     Crop the background from the foreground*",
    "  -m  --mirror=   Fill the background with a mirror of the foreground*",
    "  -0  --bgzero=   Fill the background with 0s*",
    "* The foreground is given as 4 integers (top-left and bottom-right corners) or . to automatically detect the background in each image. You can also use a file (or - for stdin) and one line will be read for each image (repeating the lsat line as necessary)."
    )

re_4_ints = "^\s*(\d+)[:;,\s]+(\d+)[:;,\s]+(\d+)[:;,\s]+(\d+)\s*$"

file_counts = { }
current_file_counts = { }
file_last_value = { }

def get_4_ints(a, help_msg):
    from re import search
    from images import Rectangle
    if a == '.': # auto-calculate
        file_counts[None] = file_counts.get(None, 0) + 1
        return None
    result = search(re_4_ints, a)
    if result == None:
        if a == '-':
            from sys import stdin
            f = stdin
        else:
            try: f = open(a, 'r')
            except: help_msg(2, "Unable to open file '%s'" % a)
        file_counts[f] = file_counts.get(f, 0) + 1
        return f
    return Rectangle(int(result.group(1)), int(result.group(2)), int(result.group(3)), int(result.group(4)))

def get_rect(im, x):
    from re import search
    from images import Rectangle
    if isinstance(x, Rectangle): return x
    c = current_file_counts.get(x, 0)
    if c == 0 or c == file_counts[x]:
        if x == None:
            from images import get_foreground_area
            r = get_foreground_area(im)
            print r
        else:
            if x.closed: r = file_last_value.get(x, None)
            else:
                l = x.readline()
                if l[-1] != '\n': x.close()
                result = search(re_4_ints, l)
                if result == None:
                    x.close()
                    r = file_last_value.get(x, None)
                else: r = Rectangle(int(result.group(1)), int(result.group(2)), int(result.group(3)), int(result.group(4)))
            if r == None: help_msg(2, "Unable to read 4 integers from '%s'" % a)
        file_last_value[x] = r
        current_file_counts[x] = 1
    else:
        r = file_last_value[x]
        current_file_counts[x] = c + 1
    return r

def parse_opt(o,a,help_msg):
    if o == "-f" or o == "--flip":
        from images import flip_up_down
        return flip_up_down
    elif o == "-F" or o == "--float":
        from images import float_image
        return float_image
    elif o == "-l" or o == "--label":
        from images import label
        return lambda im: label(im)[0]
    elif o == "-r" or o == "--relabel":
        from images import relabel
        return lambda im: relabel(im)[0]
    elif o == "-s" or o == "--sigma":
        from math import isnan
        try: sigma = float(a)
        except: help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
        if sigma < 0 or isnan(sigma): help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
        from images import gauss_blur
        return lambda im: gauss_blur(im, sigma)
    elif o == "-t" or o == "--thresh":
        if not a.isdigit() and (len(a) <= 1 or a[0] != '-' or not a[1:].isdigit()): help_msg(2, "Threshold must be an integer")
        threshold = int(a)
        from images import bw
        return lambda im: bw(im, threshold)
    elif o == "-H" or o == "--histeq":
        from images import histeq
        if a.isdigit():
            nbins = int(a)
            return lambda im: histeq(im, nbins=nbins)
        else:
            from sys import stdin
            from numpy import loadtxt
            try: histogram = loadtxt(stdin if a == '-' else a, dtype=int)
            except: help_msg(2, "Unable to load histogram data from '%s'" % a)
            return lambda im: histeq(im, hgram=histogram)
    elif o == "-p" or o == "--pad":
        padding = get_4_ints(a, help_msg)
        if padding == None: help_msg(2, "'%s' is not valid for padding" % a)
        from images import pad
        return lambda im: pad(im, *get_rect(im, padding).rect)
    elif o == "-c" or o == "--crop":
        fg = get_4_ints(a, help_msg)
        from images import crop
        return lambda im: crop(im, get_rect(im, fg))
    elif o == "-m" or o == "--mirror":
        fg = get_4_ints(a, help_msg)
        from images import fill_background
        return lambda im: fill_background(im, get_rect(im, fg), mirror=True)
    elif o == "-0" or o == "--bgzero":
        fg = get_4_ints(a, help_msg)
        from images import fill_background
        return lambda im: fill_background(im, get_rect(im, fg), 0)
    else:
        help_msg(2, "Invalid argument: "+o)

def list2imfilter(l):
    if l == None or len(l) == 0:
        return lambda im: im
    elif len(l) == 1:
        return l[0]
    else:
        def l2imf(im):
            for f in l: im = f(im)
            return im
        return l2imf
