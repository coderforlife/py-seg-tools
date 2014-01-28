getopt_short = "fFlrm:s:t:H:"
getopt_long = ["flip", "float", "label", "relabel", "sigma=", "thresh=", "histeq="]
usage = (
    "  -f  --flip      If given then each image is flipped top to bottom before saving",
    "  -F  --float     Output a floating-point image from 0 to 1",
    "  -l  --label     Output a consecutively numbered label data using connected components",
    "  -r  --relabel   Output a renumbered label image (correcting for missing or split regions)",
    "  -s  --sigma=    Sigma for Gaussian blurring while saving, defaults to no blurring",
    "  -t  --thresh=   Convert image to black and white with the given threshold (values below are 0, values above and included are 1 - reversed with negative values)",
    "  -H  --histeq=   Perform histogram equalization; if value is a number then equal-sized bins are used, otherwise it needs to be a file of integers (or - for stdin)",
    )

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
            try: histogram = loadtxt(stdin if a == '-' else a, dtype=int)
            except: help_msg(2, "Unable to load histogram data from '%s'" % a)
            return lambda im: histeq(im, hgram=histogram)
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
