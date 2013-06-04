#!/usr/bin/env python
"""
Module docstring.
"""


import os
import os.path
from sys import stderr, argv, exit

from utils import *
check_reqs()

from images import *
from process import Process

# TODO:
#  Integrate Mojtaba's code correctly (inc getting those temp files)
#  Add final step
#  Add step #1.5
#  Allow training mask to be supplied instead of training model
#  Add extra arguments for imodmop conversion
#  Do training subvolumes for speed (instead of assuming training is independent of full)

from sys import stdout
stdout = os.fdopen(stdout.fileno(), 'w', 0)
stderr = os.fdopen(stderr.fileno(), 'w', 0)


def imodmop_cmd(args, model, in_mrc, out_mrc, contract = 0):
    # TODO: support subvolume arguments
    args = ['imodmop'] + list(args)
    if contract != 0:
        args.append('-pad')
        args.append(str(-contract))
    args.append(model)
    args.append(in_mrc)
    args.append(out_mrc)
    return args

def create_color_mask_cmd(model, in_mrc, out_mrc, contract = 0, extra_args = ()):
    return imodmop_cmd(['-color', '-mask', '255'] + list(extra_args), model, in_mrc, out_mrc, contract)

def create_inv_bw_mask_cmd(model, in_mrc, out_mrc, contract = 0, extra_args = ()):
    return imodmop_cmd(['-mask', '1', '-invert', '-mode', '0'] + list(extra_args), model, in_mrc, out_mrc, contract)

# Generate Texture Dictionary Data
def genTextonDict_cmd(images, labels, output='textondict.ssv'): # images are already blurred
    if len(images) == 0 or len(images) != len(labels): raise ValueError()
    args = ['genTextonDict']
    for v, l in zip(images, labels):
        args.append('-v'); args.append(v)
        args.append('-l'); args.append(l)
    args.append(output)
    return args

# Generate Training Data
def rf_train_cmd(bcfeats, bclabels, treeNum, mtry, sampsize=0, output='bcmodel'):
    if len(bcfeats) == 0 or len(bcfeats) != len(bclabels): raise ValueError()
    args = ['rf_train']
    for f, l in zip(bcfeats, bclabels):
        args.append('-f'); args.append(f)
        args.append('-l'); args.append(l)
    args.append(treeNum)
    args.append(mtry)
    args.append(sampsize)
    args.append(output)
    return args

# Generate Predictions
def rf_predict_procs(model, features, predictions, p_bcmodel, p_features, cwd = None):
    if len(features) == 0 or len(features) != len(predictions) or len(features) != len(p_features): raise ValueError()
    base = ['rf_predict', model, '1']
    cnt = len(features)
    max = int(Process.get_max_count())
    procs = [ None ] * cnt
    if cnt < max:
        ranges = ([x] for x in xrange(cnt))
    else:
        ranges = []
        n = cnt / max
        r = cnt % max
        x = 0
        for _ in xrange(r):      ranges.append(range(x, x + n + 1, 1)); x += n + 1
        for _ in xrange(r, max): ranges.append(range(x, x + n, 1)); x += n
    for r in ranges:
        args = base[:]
        deps = [p_bcmodel,]
        for i in r:
            args.append('-f'); args.append(features[i])
            args.append('-p'); args.append(predictions[i])
            deps.append(p_features[i])
        p = Process(args, deps, cwd=cwd)
        for i in r: procs[i] = p
    return procs


def help_msg(err = 0, msg = None):
    from textwrap import fill, TextWrapper
    w = max(get_terminal_width(), 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] training.mrc training.mod full.mrc" % os.path.basename(argv[0]))
    #print "    or "
    #print tw.fill("  %s [args] full.mrc --training=x1,y1,z1,x2,y2,z2 training.mod" % os.path.basename(argv[0]))
    print ""
    print "Required arguments:"
    print tw.fill("  -w  --water-lvl The watershed water level parameter, use <=0.01") # TODO: or 0.005?
    print ""
    print "Optional algorithm parameters:"
    print tw.fill("  -c  --contract  The amount to contract contours by to make them inside the membranes")
    print tw.fill("  -S  --sigma     The amount of Gaussian blur to use, default is 1.0 while 0.0 turns off blurring")
    print tw.fill("  -n  --num-trees Number of random forest trees, should be at least 100, default is 255 - larger will be slower")
    print tw.fill("  -M  --mtry      Number of features to use in each node in RF, should be <<85, default is sqrt(total features)")
    print tw.fill("  -s  --samp-size Fraction of samples used in each node in RF, default is 0.667")
    print ""
    print "Other optional arguments:"
    print tw.fill("  -h  --help      Display this help")
    print tw.fill("  -T  --temp      Set temporary directory, default value is ./temp")
    print tw.fill("  -j  --jobs      Maximum number of jobs to do at once, default is num of processors")
    exit(err)

def die(err, msg):
    print >> stderr, msg
    exit(err)

    

if __name__ == "__main__":
    from getopt import getopt, error as getopt_error
    from glob import glob
    from math import isnan
    global max_subprocces

    argv = [argv[0], '-w0.005', 'subvolume.ali', 'run1-a.mod', 'run1-a-2.ali']

    if len(argv) < 2: help_msg(1)

    try:
        opts, args = getopt(argv[1:], "hT:j:w:c:S:n:M:s:", ["help", "jobs=", "temp=", "water-lvl=", "contract=", "sigma=", "num-trees=", "mtry=", "samp-size="])
    except getopt_error, msg: help_msg(2, msg)

    # Check the arguments
    temp = None
    jobs = None
    wl = None
    contract = None
    sigma = None
    treeNum = None
    mtry = None
    sampSize = None
    for o,a in opts:
        if o == "-h" or o == "--help":
            help_msg()
        elif o == "-T" or o == "--temp":
            if temp != None: help_msg(2, "Must be only one temp argument")
            temp = os.path.realpath(a)
        elif o == "-j" or o == "--jobs":
            if jobs != None: help_msg(2, "Must be only one jobs argument")
            if not a.isdigit() or int(a) <= 0: help_msg(2, "Number of jobs must be a positive integer")
            jobs = int(a)
##        elif o == "-z":
##            if z != None: die("Must be only one z argument", 2)
##            z = [int(s) for s in a.split(',') if s.isdigit()]
##            if len(z) != 0 or z[0] < 0 or z[1] < z[0]: help_msg(2, "The z argument must be in the form of #,# where # are non-negative integers")
        elif o == "-w" or o == "--water-lvl":
            if wl != None: help_msg(2, "Must be only one water-lvl argument")
            try: wl = float(a)
            except: help_msg(2, "Water-lvl must be a floating-point number from 0.0 to 1.0")
            if wl <= 0 or wl >= 1 or isnan(wl): help_msg(2, "Water-lvl must be a floating-point number from 0.0 to 1.0")
        elif o == "-c" or o == "--contract":
            if contract != None: help_msg(2, "Must be only one contract argument")
            try: contract = float(a)
            except: help_msg(2, "Contract must be a floating-point number")
            if isnan(contact): help_msg(2, "Contact must be a floating-point number")
        elif o == "-S" or o == "--sigma":
            if sigma != None: help_msg(2, "Must be only one sigma argument")
            try: sigma = float(a)
            except: help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
            if sigma < 0 or isnan(sigma): help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
        elif o == "-n" or o == "--num-trees":
            if treeNum != None: help_msg(2, "Must be only one num-trees argument")
            if not a.isdigit() or int(a) <= 0 or int(a) >= 40: help_msg(2, "treeNum must be a positive integer")
            treeNum = int(a)
        elif o == "-M" or o == "--mtry":
            if mtry != None: help_msg(2, "Must be only one mtry argument")
            if not a.isdigit() or int(a) <= 0 or int(a) >= 40: help_msg(2, "mtry must be a positive integer much less than 85")
            mtry = int(a)
        elif o == "-s" or o == "--samp-size":
            if sampSize != None: help_msg(2, "Must be only one samp-size argument")
            try: sampSize = float(a)
            except: help_msg(2, "Samp-size must be a floating-point number between 0.0 and 1.0")
            if sampSize <= 0 or sampSize >= 1 or isnan(sampSize): help_msg(2, "Samp-size must be a floating-point number between 0.0 and 1.0")

    # Check the MRC/MOD arguments
    if len(args) != 3: help_msg(2, "You need to provide a training MRC and MOD file along with a full dataset MRC file as arguments")
    mrc_t_filename = os.path.realpath(args[0])
    if not os.path.exists(mrc_t_filename): help_msg(2, "Training MRC file does not exist")
    mod_t_filename = os.path.realpath(args[1])
    if not os.path.exists(mod_t_filename): help_msg(2, "Training MOD file does not exist")
    mrc_f_filename = os.path.realpath(args[2])
    if not os.path.exists(mrc_f_filename): help_msg(2, "Full dataset MRC file does not exist")
    try: mrc_t = MRC(mrc_t_filename)
    except BaseException as e: help_msg(2, "Failed to open training MRC file: " + str(e))
    try: mrc_f = MRC(mrc_f_filename)
    except BaseException as e: help_msg(2, "Failed to open full dataset MRC file: " + str(e))
        
    # Check the required arguments and set defaults for optional args
    if wl       == None: help_msg(2, "water-lvl is a required argument")
    if jobs     != None: Process.set_max_count(jobs)
    if contract == None: contract = 0
    if sigma    == None: sigma = 1.0
    if treeNum  == None: treeNum = 255
    if mtry     == None: mtry = 0

    # Create temporary directory
    if temp == None: temp = os.path.realpath(os.path.join(os.getcwd(), "temp"))
    if not make_dir(temp): help_msg(2, "Temporary directory already exists as regular file, choose another directory")

    # TODO: support subvolume for speed
    zs_t = range(len(mrc_t))
    zs_f = range(len(mrc_f))
    mrc_t.close()
    mrc_f.close()


    # Notes on my conventions:
    # Since there are so many variables coming up, I starting a naming system for them
    # They begin with "f_" and "t_" to represent the full and training datasets respectively
    # The first set (during image conversions) have "d_" for raw data and "s_" for label/segmented data while the end is the file type (PNG, MHA, or MHA-blurred [simply blur])
    # Other names end with some sort of short descriptor of the contents (is = initial segmentation, tree = seg tree, sal = seg saliency, ...)
    # Later on the variables get "p_" added to the front to represent the process(es) that create those files

    ## All of the file names that will be used, relative to temporary directory
    f_d_png_folder = 'f_d_png'
    t_d_png_folder = 't_d_png'
    f_d_blur_folder = 'f_d_blur'
    t_d_blur_folder = 't_d_blur'
    t_s_bw_png_folder = 't_s_png'
    t_s_clr_mha_folder = 't_s_mha'

    f_d_png    = [('f_d_png/%03d.png'   % i) for i in zs_f] # full dataset     (PNG)
    t_d_png    = [('t_d_png/%03d.png'   % i) for i in zs_t] # training dataset (PNG)
    f_d_blur   = [('f_d_blur/%03d.mha'  % i) for i in zs_f] # full dataset     (MHA-blurred)
    t_d_blur   = [('t_d_blur/%03d.mha'  % i) for i in zs_t] # training dataset (MHA-blurred)

    t_s_bw     = 't_s_bw.mrc'  # training labels (black and white)
    t_s_clr    = 't_s_clr.mrc' # training labels (colored)
    t_s_bw_png = [('t_s_bw/%03d.png'    % i) for i in zs_t] # training labels  (PNG - black and white)
    t_s_clr_mha= [('t_s_clr/%03d.mha'   % i) for i in zs_t] # training labels  (MHA - colored)

    f_p_png_folder = 'f_p_png'
    f_p_png    = [('f_p_png/%03d.png'   % i) for i in zs_f] # full probabilty map (PNG)
    f_p_mha    = [('f_p_mha/%03d.mha'   % i) for i in zs_f] # full probabilty map (MHA)
    f_p_blur   = [('f_p_blur/%03d.mha'  % i) for i in zs_f] # full probabilty map (MHA-blurred)
    t_p_png_folder = 't_p_png'
    t_p_png    = [('t_p_png/*.%03d_cv2_float.png' % i) for i in zs_t] # TODO: full probabilty map (PNG)
    t_p_mha    = [('t_p_mha/%03d.mha'   % i) for i in zs_t] # training probabilty map (MHA)
    t_p_blur   = [('t_p_blur/%03d.mha'  % i) for i in zs_t] # training probabilty map (MHA-blurred)

    textondict = 'textondict.ssv' # Texture data

    t_is       = [('t_is/%03d.mha'      % i) for i in zs_t] # training initial segmentation
    t_tree     = [('t_tree/%03d.mha'    % i) for i in zs_t] # training segmentation tree
    t_sal      = [('t_sal/%03d.mha'     % i) for i in zs_t] # training segmentation saliency
    t_bcf      = [('t_bcf/%03d.ssv'     % i) for i in zs_t] # training segmentation features
    t_bcl      = [('t_bcl/%03d.ssv'     % i) for i in zs_t] # training segmentation labels

    bcmodel = 'bcmodel' # Training data

    f_is       = [('f_is/%03d.mha'      % i) for i in zs_f] # full initial segmentation
    f_tree     = [('f_tree/%03d.mha'    % i) for i in zs_f] # full segmentation tree
    f_sal      = [('f_sal/%03d.mha'     % i) for i in zs_f] # full segmentation saliency
    f_bcf      = [('f_bcf/%03d.ssv'     % i) for i in zs_f] # full segmentation features
    f_bcp      = [('f_bcp/%03d.ssv'     % i) for i in zs_f] # full segmentation predictions

    seg_mha    = [('seg/%03d.mha'       % i) for i in zs_f] # TODO: 

    ### Convert input files ###
    p_f_d_png    = Process(('mrc2png', mrc_f_filename, f_d_png_folder), cwd=temp)
    p_t_d_png    = Process(('mrc2png', mrc_t_filename, t_d_png_folder), cwd=temp)

    p_f_d_blur   = Process(('mrc2mha', '-mfloat', '-s'+str(sigma), mrc_f_filename, f_d_blur_folder), cwd=temp)
    p_t_d_blur   = Process(('mrc2mha', '-mfloat', '-s'+str(sigma), mrc_t_filename, t_d_blur_folder), cwd=temp)

    p_t_s_bw     = Process(create_inv_bw_mask_cmd(mod_t_filename, mrc_t_filename, t_s_bw, contract), cwd=temp) # TODO: support extra args
    p_t_s_clr    = Process(create_color_mask_cmd(mod_t_filename, mrc_t_filename, t_s_clr, contract), cwd=temp) # TODO: support extra args

    p_t_s_bw_png = Process(('mrc2png', t_s_bw, t_s_bw_png_folder), (p_t_s_bw,), cwd=temp)
    p_t_s_clr_mha= Process(('mrc2mha', '-mlabel', t_s_clr, t_s_clr_mha_folder), (p_t_s_clr,), cwd=temp)


    ### Generate membrane segmentation from Mojtaba's code and convert resulting files ###
    p_p_png      = Process(('moj-seg', t_d_png_folder, t_s_bw_png_folder, f_d_png_folder, f_p_png_folder, t_p_png_folder), (p_t_d_png, p_f_d_png, p_t_s_bw_png), cwd=temp)
    p_f_p_mha    = [Process(('png2mha', '-mfloat', png, mha), (p_p_png,), cwd=temp) for png, mha in zip(f_p_png, f_p_mha)]
    p_t_p_mha    = [Process(('png2mha', '-mfloat', png, mha), (p_p_png,), cwd=temp) for png, mha in zip(t_p_png, t_p_mha)]
    if sigma == 0.0:
        p_f_p_blur = [Process(('cp', png, mha), (ppm,), cwd=temp) for png, mha, ppm in zip(f_p_mha, f_p_blur, p_f_p_mha)]
        p_t_p_blur = [Process(('cp', png, mha), (ppm,), cwd=temp) for png, mha, ppm in zip(t_p_mha, t_p_blur, p_t_p_mha)]
    else:
        p_f_p_blur = [Process(('png2mha', '-mfloat', '-s'+str(sigma), png, mha), (p_p_png,), cwd=temp) for png, mha in zip(f_p_png, f_p_blur)]
        p_t_p_blur = [Process(('png2mha', '-mfloat', '-s'+str(sigma), png, mha), (p_p_png,), cwd=temp) for png, mha in zip(t_p_png, t_p_blur)]


    ### Training Phase ###
    # 0 - Training texures
    p_textondict = Process(genTextonDict_cmd(t_d_blur, t_s_clr_mha, textondict), (p_t_s_clr_mha, p_t_d_blur), cwd=temp)
    # 1 - Training initial segmentation
    p_t_is       = [Process(('watershed', pb, wl, iseg), (ppb,), cwd=temp)
                    for pb, iseg, ppb in zip(t_p_blur, t_is, p_t_p_blur)]
    # TODO: add step 1.5
    # 2 - Training merge generation
    p_t_merge    = [Process(('genMerges', iseg, pb, t, s), (pis,), cwd=temp) # + p_t_p_blur
                    for iseg, pb, t, s, pis in zip(t_is, t_p_blur, t_tree, t_sal, p_t_is)]
    # 3 - Training boundary feature generation
    p_t_bfeat    = [Process(('genBoundaryFeatures', iseg, t, s, db, p, textondict, '0', bcf), (p_textondict, ppm, pm), cwd=temp) # + p_t_is, p_t_d_blur
                    for iseg, t, s, db, p, bcf, ppm, pm in zip(t_is, t_tree, t_sal, t_d_blur, t_p_mha, t_bcf, p_t_p_mha, p_t_merge)]
    # 4 - Training boundary label generation
    p_t_blbl     = [Process(('genBoundaryLabels', iseg, t, l, bcl), (p_t_s_clr_mha, pm), cwd=temp) # + p_t_is
                    for iseg, t, l, bcl, pm in zip(t_is, t_tree, t_s_clr_mha, t_bcl, p_t_merge)]
    # 5 - Training Data Generation
    p_bcmodel    = Process(rf_train_cmd(t_bcf, t_bcl, treeNum, mtry, sampSize, bcmodel), p_t_bfeat + p_t_blbl, cwd=temp)


    ### Segmentation Phase ###
    # 1 - Full dataset initial segmentation
    p_f_is       = [Process(('watershed', pb, wl, iseg), (ppb,), cwd=temp)
                    for pb, iseg, ppb in zip(f_p_blur, f_is, p_f_p_blur)]
    # TODO: add step 1.5
    # 2 - Full dataset merge generation
    p_f_merge    = [Process(('genMerges', iseg, pb, t, s), (pis,), cwd=temp) # + p_f_p_blur
                    for iseg, pb, t, s, pis in zip(f_is, f_p_blur, f_tree, f_sal, p_f_is)]
    # 3 - Full dataset boundary feature generation
    p_f_bfeat    = [Process(('genBoundaryFeatures', iseg, t, s, db, p, textondict, '0', bcf), (p_textondict, p_f_d_blur, ppm, pm), cwd=temp) # + p_f_is
                    for iseg, t, s, db, p, bcf, ppm, pm in zip(f_is, f_tree, f_sal, f_d_blur, f_p_mha, f_bcf, p_f_p_mha, p_f_merge)]
    # 6 - Generate Predictions
    p_f_bcp      = rf_predict_procs(bcmodel, f_bcf, f_bcp, p_bcmodel, p_f_bfeat, temp)
    # 7 - Segment
    # TODO: output format (the 1 and 0) is currently set to PNG black/white (0/1 would be MHA numbered)
    p_seg_mha    = [Process(('segment', iseg, t, bcp, '0', '1', '0', 'NULL', s), (pbcp,), cwd=temp) # + p_f_is, p_f_merge
                    for iseg, t, bcp, s, pbcp in zip(f_is, f_tree, f_bcp, seg_mha, p_f_bcp)]

    ### Final Conversion ###
    # TODO
    p = Process(('TODO: final conversion'), (p_seg_mha), cwd=temp)


    p.run()
