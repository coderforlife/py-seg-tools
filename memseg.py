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
from tasks import Tasks

# TODO:
#  Add final step
#  Add step #1.5
#  Make smart-restart
#  Allow training mask to be supplied instead of training model
#  Add extra arguments for imodmop conversion
#  Do training subvolumes for speed (instead of assuming training is independent of full)

from sys import stdout
if hasattr(stdout, 'fileno'):
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
def rf_predict_procs(proc, model, features, predictions):
    if len(features) == 0 or len(features) != len(predictions): raise ValueError()
    base = ['rf_predict', model, '1']
    cnt = len(features)
    max = int(Tasks.get_max_at_once())
    if cnt < max:
        ranges = ([x] for x in xrange(cnt))
    else:
        ranges, n, r, x = [], cnt / max, cnt % max, 0
        for _ in xrange(r):      ranges.append(range(x, x + n + 1, 1)); x += n + 1
        for _ in xrange(r, max): ranges.append(range(x, x + n, 1)); x += n
    for r in ranges:
        args = base[:]
        inputs = [model,]
        outputs = []
        for i in r:
            args.append('-f'); args.append(features[i])
            args.append('-p'); args.append(predictions[i])
            inputs.append(features[i])
            outputs.append(predictions[i])
        proc.add(args, inputs, outputs)


def help_msg(err = 0, msg = None):
    from textwrap import fill, TextWrapper
    w = max(get_terminal_width(), 20)
    tw = TextWrapper(width = w, subsequent_indent = ' '*18)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] training.mrc training.mod full.mrc output.mod" % os.path.basename(argv[0]))
    #print "    or "
    #print tw.fill("  %s [args] full.mrc --training=x1,y1,z1,x2,y2,z2 training.mod output.mod" % os.path.basename(argv[0]))
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

    #argv = [argv[0], '-w0.005', 'training.mrc', 'training.mod', 'full.mrc', 'output.mod']

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
            if not a.isdigit() or int(a) == 0: help_msg(2, "Number of jobs must be a positive integer")
            jobs = int(a)
##        elif o == "-z":
##            if z != None: die("Must be only one z argument", 2)
##            z = [int(s) for s in a.split(',') if s.isdigit()]
##            if len(z) != 0 or z[1] < z[0]: help_msg(2, "The z argument must be in the form of #,# where # are non-negative integers")
        elif o == "-w" or o == "--water-lvl":
            if wl != None: help_msg(2, "Must be only one water-lvl argument")
            try: wl = float(a)
            except: help_msg(2, "Water-lvl must be a floating-point number from 0.0 to 1.0")
            if wl <= 0 or wl >= 1 or isnan(wl): help_msg(2, "Water-lvl must be a floating-point number from 0.0 to 1.0")
        elif o == "-c" or o == "--contract":
            if contract != None: help_msg(2, "Must be only one contract argument")
            try: contract = float(a)
            except: help_msg(2, "Contract must be a floating-point number")
            if isnan(contract): help_msg(2, "Contract must be a floating-point number")
        elif o == "-S" or o == "--sigma":
            if sigma != None: help_msg(2, "Must be only one sigma argument")
            try: sigma = float(a)
            except: help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
            if sigma < 0 or isnan(sigma): help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
        elif o == "-n" or o == "--num-trees":
            if treeNum != None: help_msg(2, "Must be only one num-trees argument")
            if not a.isdigit() or int(a) == 0: help_msg(2, "treeNum must be a positive integer")
            treeNum = int(a)
        elif o == "-M" or o == "--mtry":
            if mtry != None: help_msg(2, "Must be only one mtry argument")
            if not a.isdigit() or not (0 < int(a) < 40): help_msg(2, "mtry must be a positive integer much less than 85")
            mtry = int(a)
        elif o == "-s" or o == "--samp-size":
            if sampSize != None: help_msg(2, "Must be only one samp-size argument")
            try: sampSize = float(a)
            except: help_msg(2, "Samp-size must be a floating-point number between 0.0 and 1.0")
            if not (0 <= sampSize <= 1) or isnan(sampSize): help_msg(2, "Samp-size must be a floating-point number between 0.0 and 1.0")

    # Check the MRC/MOD arguments
    if len(args) != 4: help_msg(2, "You need to provide a training MRC and MOD file along with a full dataset MRC file as arguments")
    mrc_t_filename = os.path.realpath(args[0])
    if not os.path.exists(mrc_t_filename): help_msg(2, "Training MRC file does not exist")
    mod_t_filename = os.path.realpath(args[1])
    if not os.path.exists(mod_t_filename): help_msg(2, "Training MOD file does not exist")
    mrc_f_filename = os.path.realpath(args[2])
    if not os.path.exists(mrc_f_filename): help_msg(2, "Full dataset MRC file does not exist")
    mod_output = os.path.realpath(args[3])
    if os.path.exists(mod_output) and os.path.isdir(mod_output): help_msg(2, "Output MOD file exists and is a directory")
    try: mrc_t = MRC(mrc_t_filename)
    except BaseException as e: help_msg(2, "Failed to open training dataset MRC file: " + str(e))
    try: mrc_f = MRC(mrc_f_filename)
    except BaseException as e: help_msg(2, "Failed to open full dataset MRC file: " + str(e))
        
    # Check the required arguments and set defaults for optional args
    if wl       == None: help_msg(2, "water-lvl is a required argument")
    if jobs     != None: Tasks.set_max_at_once(jobs)
    if contract == None: contract = 0
    if sigma    == None: sigma = 1.0
    if treeNum  == None: treeNum = 255
    if mtry     == None: mtry = 0
    if sampSize == None: sampSize = 0.66666666666667

    # Create temporary directory (and make paths relative)
    if temp == None: temp = os.path.realpath(os.path.join(os.getcwd(), "temp"))
    if not make_dir(temp): help_msg(2, "Temporary directory already exists as regular file, choose another directory")
    # TODO: make paths somewhat relative - os.path.commonprefix
    mrc_t_filename = os.path.relpath(mrc_t_filename, temp)
    mod_t_filename = os.path.relpath(mod_t_filename, temp)
    mrc_f_filename = os.path.relpath(mrc_f_filename, temp)
    mod_output = os.path.relpath(mod_output, temp)
    

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

    ## All of the file names that will be used, relative to temporary directory
    f_d_png_folder = 'f_d_png'
    t_d_png_folder = 't_d_png'
    f_d_png    = [('f_d_png/%03d.png'   % i) for i in zs_f] # full dataset     (PNG)
    t_d_png    = [('t_d_png/%03d.png'   % i) for i in zs_t] # training dataset (PNG)
    f_d_blur_folder = 'f_d_blur'
    t_d_blur_folder = 't_d_blur'
    f_d_blur   = [('f_d_blur/%03d.mha'  % i) for i in zs_f] # full dataset     (MHA-blurred)
    t_d_blur   = [('t_d_blur/%03d.mha'  % i) for i in zs_t] # training dataset (MHA-blurred)

    t_s_bw     = 't_s_bw.mrc'  # training labels (black and white)
    t_s_clr    = 't_s_clr.mrc' # training labels (colored)
    t_s_bw_png_folder = 't_s_bw'
    t_s_clr_mha_folder = 't_s_clr'
    t_s_bw_png = [('t_s_bw/%03d.png'    % i) for i in zs_t] # training labels  (PNG - black and white)
    t_s_clr_mha= [('t_s_clr/%03d.mha'   % i) for i in zs_t] # training labels  (MHA - colored)

    f_p_png_folder = 'f_p_png'
    f_p_png    = [('f_p_png/%03d.png'   % i) for i in zs_f] # full probabilty map (PNG)
    f_p_mha    = [('f_p_mha/%03d.mha'   % i) for i in zs_f] # full probabilty map (MHA)
    f_p_blur   = [('f_p_blur/%03d.mha'  % i) for i in zs_f] # full probabilty map (MHA-blurred)
    t_p_png_folder = 't_p_png'
    t_p_png    = [('t_p_png/%03d_cv2_float.png' % i) for i in zs_t] # TODO: full probabilty map (PNG)
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

    seg_pts    = [('seg_pts/%03d.pts'   % i) for i in zs_f] # the final segementation points for each section
    seg_pts_folder = 'seg_pts'
    seg_pts_all = 'segmentation.pts' # the final segementation points for all sections


    ### Clean out temporary directories ###
    # TODO: support this and smart-restart
    #clear_dir(f_d_png_folder, "*.png")
    #clear_dir(t_s_bw_png_folder, ".png")
    #clear_dir(t_d_png_folder, "*.png")
    #clear_dir(f_p_png_folder, "*.png")
    #clear_dir(t_p_png_folder) # temp directory has lots of stuff in it

    memseg = Tasks('memseg.log',
                   {'waterlevel':wl,'contract':contract,'sigma':sigma,'number-of-trees':treeNum,'mtry':mtry,'sample-size':sampSize},
                   workingdir = temp)

    ### Convert input files ###
    memseg.add(('mrc2png', mrc_f_filename, f_d_png_folder), (mrc_f_filename,), f_d_png)
    memseg.add(('mrc2png', mrc_t_filename, t_d_png_folder), (mrc_t_filename,), t_d_png)

    memseg.add(('mrc2mha', '-mfloat', '-s'+str(sigma), mrc_f_filename, f_d_blur_folder), (mrc_f_filename,), f_d_blur, ('sigma',))
    memseg.add(('mrc2mha', '-mfloat', '-s'+str(sigma), mrc_t_filename, t_d_blur_folder), (mrc_t_filename,), t_d_blur, ('sigma',))

    memseg.add(create_inv_bw_mask_cmd(mod_t_filename, mrc_t_filename, t_s_bw,  contract), (mod_t_filename, mrc_t_filename), (t_s_bw, ), ('contract',)) # TODO: support extra args
    memseg.add(create_color_mask_cmd (mod_t_filename, mrc_t_filename, t_s_clr, contract), (mod_t_filename, mrc_t_filename), (t_s_clr,), ('contract',)) # TODO: support extra args

    memseg.add(('mrc2png',            t_s_bw,  t_s_bw_png_folder ), (t_s_bw, ), t_s_bw_png )
    memseg.add(('mrc2mha', '-mlabel', t_s_clr, t_s_clr_mha_folder), (t_s_clr,), t_s_clr_mha)


    ### Generate membrane segmentation from Mojtaba's code and convert resulting files ###
    memseg.add(('moj-seg', t_d_png_folder, t_s_bw_png_folder, f_d_png_folder, f_p_png_folder, t_p_png_folder), t_d_png+t_s_bw_png+f_d_png, f_p_png+t_p_png)
    [memseg.add(('png2mha', '-mfloat', png, mha), (png,), (mha,)) for png, mha in zip(f_p_png, f_p_mha)]
    [memseg.add(('png2mha', '-mfloat', png, mha), (png,), (mha,)) for png, mha in zip(t_p_png, t_p_mha)]
    if sigma == 0.0:
        [memseg.add(('cp', png, mha), (png,), (mha,), ('sigma',)) for png, mha in zip(f_p_mha, f_p_blur)]
        [memseg.add(('cp', png, mha), (png,), (mha,), ('sigma',)) for png, mha in zip(t_p_mha, t_p_blur)]
    else:
        [memseg.add(('png2mha', '-mfloat', '-s'+str(sigma), png, mha), (png,), (mha,), ('sigma',)) for png, mha in zip(f_p_png, f_p_blur)]
        [memseg.add(('png2mha', '-mfloat', '-s'+str(sigma), png, mha), (png,), (mha,), ('sigma',)) for png, mha in zip(t_p_png, t_p_blur)]


    ### Training Phase ###
    # 0 - Training texures
    memseg.add(genTextonDict_cmd(t_d_blur, t_s_clr_mha, textondict), t_d_blur+t_s_clr_mha, (textondict,))
    # 1 - Training initial segmentation
    [memseg.add(('watershed', pb, wl, iseg), (pb,), (iseg,), ('waterlevel',)) for pb, iseg in zip(t_p_blur, t_is)]
    # TODO: add step 1.5
    # 2 - Training merge generation
    [memseg.add(('genMerges', iseg, pb, t, s), (iseg, pb), (t, s)) for iseg, pb, t, s in zip(t_is, t_p_blur, t_tree, t_sal)]
    # 3 - Training boundary feature generation
    [memseg.add(('genBoundaryFeatures', iseg, t, s, db, p, textondict, '0', bcf), (iseg, t, s, db, p, textondict), (bcf,)) for iseg, t, s, db, p, bcf in zip(t_is, t_tree, t_sal, t_d_blur, t_p_mha, t_bcf)]
    # 4 - Training boundary label generation
    [memseg.add(('genBoundaryLabels', iseg, t, l, bcl), (iseg, t, l), (bcl,)) for iseg, t, l, bcl in zip(t_is, t_tree, t_s_clr_mha, t_bcl)]
    # 5 - Training Data Generation
    memseg.add(rf_train_cmd(t_bcf, t_bcl, treeNum, mtry, sampSize, bcmodel), t_bcf+t_bcl, (bcmodel,), ('number-of-trees','mtry','sample-size'))


    ### Segmentation Phase ###
    # 1 - Full dataset initial segmentation
    [memseg.add(('watershed', pb, wl, iseg), (pb,), (iseg,), ('waterlevel',)) for pb, iseg in zip(f_p_blur, f_is)]
    # TODO: add step 1.5
    # 2 - Full dataset merge generation
    [memseg.add(('genMerges', iseg, pb, t, s), (iseg, pb), (t, s)) for iseg, pb, t, s in zip(f_is, f_p_blur, f_tree, f_sal)]
    # 3 - Full dataset boundary feature generation
    [memseg.add(('genBoundaryFeatures', iseg, t, s, db, p, textondict, '0', bcf), (iseg, t, s, db, p, textondict), (bcf,)) for iseg, t, s, db, p, bcf in zip(f_is, f_tree, f_sal, f_d_blur, f_p_mha, f_bcf)]
    # 6 - Generate Predictions
    rf_predict_procs(memseg, bcmodel, f_bcf, f_bcp)
    # 7 - Segment
    [memseg.add(('segmentToContours', iseg, t, bcp, z, sp), (iseg, t, bcp), (sp,)) for iseg, t, bcp, z, sp in zip(f_is, f_tree, f_bcp, zs_f, seg_pts)]
    

    ### Convert output files ###
    memseg.add(['combine_points',] + seg_pts + [seg_pts_all,], seg_pts, (seg_pts_all,))
    # TODO: -im and pixel spacing?
    memseg.add(('point2model', '-im', mrc_f_filename, seg_pts_all, mod_output), (mrc_f_filename,seg_pts_all), (mod_output,))


    # Run!
    memseg.run(verbose = True)
