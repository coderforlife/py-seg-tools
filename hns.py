#!/usr/bin/env python
"""
Module docstring.
"""

from utils import *
check_reqs(psutil = True)

from images import *
from mrc import *
from tasks import *

# TODO:
#  Verify final step
#  Make water-level optional and automatically found
#  Allow training mask to be supplied instead of training model
#  Add extra arguments for imodmop conversion
#  Do training subvolumes for speed (instead of assuming training is independent of full)

import os
from sys import stdout, stderr, argv, exit
if hasattr(stdout, 'fileno'):
    try:
        stdout = os.fdopen(stdout.fileno(), 'w', 0)
        stderr = os.fdopen(stderr.fileno(), 'w', 0)
    except: pass


def imodmop_cmd(args, model, in_mrc, out_mrc, contract = 0):
    # TODO: support subvolume arguments
    args = ['imodmop'] + list(args)
    if contract != 0:
        args.append('-pad')
        args.append(-contract)
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
    from itertools import izip

    if len(images) == 0 or len(images) != len(labels): raise ValueError()
    args = ['hnsGenTextonDict']
    for v, l in izip(images, labels):
        args.append('-v'); args.append(v)
        args.append('-l'); args.append(l)
    #TODO: Defaults used:
    #   maxIteration      -> 10000
    #   convergeThreshold -> 0.0002
    args.append(output)
    return args

# Generate Training Data
def rf_train_cmd(bcfeats, bclabels, treeNum, mtry, sampsize=0, output='bcmodel'):
    from itertools import izip

    if len(bcfeats) == 0 or len(bcfeats) != len(bclabels): raise ValueError()
    args = ['rf_train']
    for f, l in izip(bcfeats, bclabels):
        args.append('-f'); args.append(f)
        args.append('-l'); args.append(l)
    args.append(treeNum)
    args.append(mtry)
    args.append(sampsize)
    args.append('1') # isBalancedClass - should always be true (1)
    args.append(output)
    return args

# Generate Predictions
def rf_predict_procs(proc, model, features, predictions):
    if len(features) == 0 or len(features) != len(predictions): raise ValueError()
    base = ['rf_predict', model, '1']
    cnt = len(features)
    max = proc.max_tasks_at_once
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
    from os.path import basename
    
    w = max(get_terminal_width(), 40)
    tw = TextWrapper(width = w, subsequent_indent = ' '*20)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] training.mrc training.mod full.mrc output.mod" % basename(argv[0]))
    #print "    or "
    #print tw.fill("  %s [args] full.mrc --training=x1,y1,z1,x2,y2,z2 training.mod output.mod" % basename(argv[0]))
    print ""
    print "Required arguments:"
    print tw.fill("  -w  --water-lvl=  The watershed water level parameter, probably <=0.02")
    print ""
    print "Optional algorithm parameters:"
    print tw.fill("  -c  --contract=   The amount to contract contours by to make them inside the membranes")
    print tw.fill("  -s  --sigma=      The amount of Gaussian blur to use, default is 1.0 while 0.0 turns off blurring")
    print tw.fill("  -n  --num-trees=  Number of random forest trees, should be at least 100, default is 255 - larger will be slower")
    print tw.fill("  -m  --mtry=       Number of features to use in each node in RF, should be <<85, default is sqrt(total features)")
    print tw.fill("  -S  --samp-size=  Fraction of samples used in each node in RF, default is 0.7")
    print tw.fill("  -pm-area-thresh0= Pre-merge area threshold #1, default is 50")
    print tw.fill("  -pm-area-thresh1= Pre-merge area threshold #2, default is 200")
    print tw.fill("  -pm-prop-thresh=  Pre-merge average probability threshold, default is 0.5")
    print ""
    print "Other optional arguments:"
    print tw.fill("  -h  --help       Display this help")
    print tw.fill("  -t  --temp=      Set temporary directory, default value is ./temp")
    print tw.fill("  -j  --jobs=      Maximum number of jobs to do at once, default is num of processors")
    print tw.fill("  -u  --rusage=    Save the resources usage (memory and time) for each run process to a file [not available on Windows]")
    exit(err)

def die(err, msg):
    print >> stderr, msg
    exit(err)

    

if __name__ == "__main__":
    from getopt import getopt, error as getopt_error
    from glob import glob
    from os.path import exists, isdir, join, realpath, relpath
    from os import getcwd
    from math import isnan
    from itertools import izip

    #argv = [argv[0], '-w0.01', 'training.mrc', 'training.mod', 'full.mrc', 'output.mod']

    if len(argv) < 2: help_msg(1)

    try:
        opts, args = getopt(argv[1:], "ht:j:u:w:c:s:n:m:S:",
                            ["help", "temp=", "jobs=", "rusage=",
                             "water-lvl=", "contract=", "sigma=", "num-trees=", "mtry=", "samp-size=", "pm-area-thresh0=", "pm-area-thresh1=", "pm-prop-thresh="
                             ])
    except getopt_error, msg: help_msg(2, msg)

    # Check the arguments
    temp = None
    jobs = None
    rusage_log = None
    wl = None
    contract = None
    sigma = None
    treeNum = None
    mtry = None
    sampSize = None
    areaThreshold0 = None
    areaThreshold1 = None
    probThreshold = None
    for o,a in opts:
        if o == "-h" or o == "--help":
            help_msg()
        elif o == "-t" or o == "--temp":
            if temp != None: help_msg(2, "Must be only one temp argument")
            temp = realpath(a)
        elif o == "-j" or o == "--jobs":
            if jobs != None: help_msg(2, "Must be only one jobs argument")
            if not a.isdigit() or int(a) == 0: help_msg(2, "Number of jobs must be a positive integer")
            jobs = int(a)
        elif o == "-u" or o == "--rusage":
            if rusage_log != None: help_msg(2, "Must be only one rusage argument")
            try:
                from os_ext import wait4 # make sure wait4 is available
                rusage_log = realpath(a)
            except ImportError:
                print >> stderr, "Warning: System does not support recording resource usage, rusage argument ignored."
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
        elif o == "-s" or o == "--sigma":
            if sigma != None: help_msg(2, "Must be only one sigma argument")
            try: sigma = float(a)
            except: help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
            if sigma < 0 or isnan(sigma): help_msg(2, "Sigma must be a floating-point number greater than or equal to 0.0")
        elif o == "-n" or o == "--num-trees":
            if treeNum != None: help_msg(2, "Must be only one num-trees argument")
            if not a.isdigit() or int(a) == 0: help_msg(2, "num-trees must be a positive integer")
            treeNum = int(a)
        elif o == "-m" or o == "--mtry":
            if mtry != None: help_msg(2, "Must be only one mtry argument")
            if not a.isdigit() or not (0 < int(a) < 40): help_msg(2, "mtry must be a positive integer much less than 85")
            mtry = int(a)
        elif o == "-S" or o == "--samp-size":
            if sampSize != None: help_msg(2, "Must be only one samp-size argument")
            try: sampSize = float(a)
            except: help_msg(2, "samp-size must be a floating-point number between 0.0 and 1.0")
            if not (0 <= sampSize <= 1) or isnan(sampSize): help_msg(2, "samp-size must be a floating-point number between 0.0 and 1.0")
        elif o == "--pm-area-thresh0=":
            if areaThreshold0 != None: help_msg(2, "Must be only one pm-area-thresh0 argument")
            if not a.isdigit() or 0 >= int(a): help_msg(2, "pm-area-thresh0 must be a positive integer")
            areaThreshold0 = int(a)
        elif o == "--pm-area-thresh1=":
            if areaThreshold1 != None: help_msg(2, "Must be only one pm-area-thresh1 argument")
            if not a.isdigit() or 0 >= int(a): help_msg(2, "pm-area-thresh1 must be a positive integer")
            areaThreshold1 = int(a)
        elif o == "--pm-prop-thresh=":
            if probThreshold != None: help_msg(2, "Must be only one pm-prop-thresh argument")
            try: probThreshold = float(a)
            except: help_msg(2, "pm-prop-thresh must be a floating-point number between 0.0 and 1.0")
            if not (0 <= probThreshold <= 1) or isnan(probThreshold): help_msg(2, "pm-prop-thresh must be a floating-point number between 0.0 and 1.0")

    # Check the MRC/MOD arguments
    if len(args) != 4: help_msg(2, "You need to provide a training MRC and MOD file along with a full dataset MRC file as arguments")
    mrc_t_filename = realpath(args[0])
    if not exists(mrc_t_filename): help_msg(2, "Training MRC file does not exist")
    mod_t_filename = realpath(args[1])
    if not exists(mod_t_filename): help_msg(2, "Training MOD file does not exist")
    mrc_f_filename = realpath(args[2])
    if not exists(mrc_f_filename): help_msg(2, "Full dataset MRC file does not exist")
    mod_output = realpath(args[3])
    if exists(mod_output) and isdir(mod_output): help_msg(2, "Output MOD file exists and is a directory")
    try: mrc_t = MRC(mrc_t_filename)
    except BaseException as e: help_msg(2, "Failed to open training dataset MRC file: " + str(e))
    try: mrc_f = MRC(mrc_f_filename)
    except BaseException as e: help_msg(2, "Failed to open full dataset MRC file: " + str(e))
        
    # Check the required arguments and set defaults for optional args
    if wl       == None: help_msg(2, "water-lvl is a required argument")
    #if jobs     != None: ... # dealt with later
    #if rusage_log == None: # None means no log
    if contract == None: contract = 0
    if sigma    == None: sigma = 1.0
    if treeNum  == None: treeNum = 255
    if mtry     == None: mtry = 0 # will make the program calculate the proper default
    if sampSize == None: sampSize = 0.70
    if areaThreshold0 == None: areaThreshold0 = 50
    if areaThreshold1 == None: areaThreshold1 = 200
    if probThreshold  == None: probThreshold  = 0.50

    # Create temporary directory (and make paths relative)
    if temp == None: temp = realpath(join(getcwd(), "temp"))
    if not make_dir(temp): help_msg(2, "Temporary directory already exists as regular file, choose another directory")
    # TODO: make paths somewhat relative - os.path.commonprefix
    mrc_t_filename = relpath(mrc_t_filename, temp)
    mod_t_filename = relpath(mod_t_filename, temp)
    mrc_f_filename = relpath(mrc_f_filename, temp)
    mod_output = relpath(mod_output, temp)
    
    # Get properties from the MRCs then close them
    # TODO: support subvolume for speed
    pxls_t  = mrc_t.section_size
    bytes_t = mrc_t.section_full_data_size
    pxls_f  = mrc_f.section_size
    bytes_f = mrc_f.section_full_data_size
    zs_t = range(len(mrc_t))
    zs_f = range(len(mrc_f))
    mrc_t.close()
    mrc_f.close()

    nrounds = 2 # number of Mojtaba's segmentation rounds, only values 1-9 work
    rounds = range(1, nrounds + 1)

    # Generic filenames
    pngs_t = ['%04d.png' % i for i in zs_t]
    pngs_f = ['%04d.png' % i for i in zs_f]
    mhas_t = ['%04d.mha' % i for i in zs_t]
    mhas_f = ['%04d.mha' % i for i in zs_f]
    ssvs_t = ['%04d.ssv' % i for i in zs_t]
    ssvs_f = ['%04d.ssv' % i for i in zs_f]


    # Notes on my conventions:
    # Since there are so many variables coming up, I starting a naming system for them
    # They begin with "f_" and "t_" to represent the full and training datasets respectively
    # The first set (during image conversions) have "d_" for raw data, "s_" for label/segmented data, and "p_" for probability maps
    # The end is the file type (PNG, MHA, or MHA-blurred [simply blur])
    # Other names end with some sort of short descriptor of the contents (is = initial segmentation, tree = seg tree, sal = seg saliency, ...)

    ## All of the file names that will be used, relative to temporary directory    
    t_d_png_folder = 't_d_png'
    f_d_png_folder = 'f_d_png'
    t_d_blur_folder = 't_d_blur'
    f_d_blur_folder = 'f_d_blur'
    t_d_png    = [join(t_d_png_folder,  i) for i in pngs_t] # training dataset (PNG)
    f_d_png    = [join(f_d_png_folder,  i) for i in pngs_f] # full dataset     (PNG)
    t_d_blur   = [join(t_d_blur_folder, i) for i in mhas_t] # training dataset (MHA-blurred)
    f_d_blur   = [join(f_d_blur_folder, i) for i in mhas_f] # full dataset     (MHA-blurred)

    t_s_bw     = 't_s_bw.mrc'  # training labels (black and white)
    t_s_clr    = 't_s_clr.mrc' # training labels (colored)
    t_s_bw_png_folder = 't_s_bw'
    t_s_clr_mha_folder = 't_s_clr'
    t_s_bw_png = [join(t_s_bw_png_folder,  i) for i in pngs_t] # training labels (PNG-black and white)
    t_s_clr_mha= [join(t_s_clr_mha_folder, i) for i in mhas_t] # training labels (MHA-colored)

    t_p_png_folder = 't_p_png'
    t_p_png_temp   = [[join(t_p_png_folder, '%04d_cv%d_float.png' % (i, r)) for i in zs_t] for r in xrange(1, nrounds + 1)]
    t_p_png_temp.insert(0, [])
    t_p_png    = t_p_png_temp[nrounds]                 # training probabilty map (PNG)
    t_p_mha    = [join('t_p_mha',  i) for i in mhas_t] # training probabilty map (MHA)
    t_p_blur   = [join('t_p_blur', i) for i in mhas_t] # training probabilty map (MHA-blurred)

    f_p_png_folder = 'f_p_png'
    f_p_png_temp   = [[join(f_p_png_folder, '%04d_cv%d_float.png' % (i, r)) for i in zs_f] for r in xrange(1, nrounds + 1)]
    f_p_png_temp.insert(0, [])
    f_p_png    = f_p_png_temp[nrounds]                 # full probabilty map (PNG)
    f_p_mha    = [join('f_p_mha',  i) for i in mhas_f] # full probabilty map (MHA)
    f_p_blur   = [join('f_p_blur', i) for i in mhas_f] # full probabilty map (MHA-blurred)

    textondict = 'textondict.ssv' # Texture data

    t_is1      = [join('t_is1',  i) for i in mhas_t] # training initial segmentation (from watershed)
    t_is2      = [join('t_is2',  i) for i in mhas_t] # training initial segmentation (from pre-merging)
    t_tree     = [join('t_tree', i) for i in mhas_t] # training segmentation tree
    t_sal      = [join('t_sal',  i) for i in mhas_t] # training segmentation saliency
    t_bcf      = [join('t_bcf',  i) for i in ssvs_t] # training segmentation features
    t_bcl      = [join('t_bcl',  i) for i in ssvs_t] # training segmentation labels

    bcmodel = 'bcmodel' # Training data

    f_is1      = [join('f_is1',  i) for i in mhas_f] # full initial segmentation (from watershed)
    f_is2      = [join('f_is2',  i) for i in mhas_f] # full initial segmentation (from pre-merging)
    f_tree     = [join('f_tree', i) for i in mhas_f] # full segmentation tree
    f_sal      = [join('f_sal',  i) for i in mhas_f] # full segmentation saliency
    f_bcf      = [join('f_bcf',  i) for i in ssvs_f] # full segmentation features
    f_bcp      = [join('f_bcp',  i) for i in ssvs_f] # full segmentation predictions
    f_fs       = [join('f_fs',   i) for i in mhas_f] # the final segementation

    seg_pts_folder = 'seg_pts'
    seg_pts    = [join(seg_pts_folder,'%04d.pts' % i) for i in zs_f] # the final segementation points for each section
    seg_pts_all = 'segmentation.pts' # the final segementation points for all sections

    # All folders that are used
    folders = [
            t_d_png_folder, f_d_png_folder, t_d_blur_folder, f_d_blur_folder, t_s_bw_png_folder, t_s_clr_mha_folder,
            t_p_png_folder, 't_p_mha', 't_p_blur',
            f_p_png_folder, 'f_p_mha', 'f_p_blur',
            't_is1', 't_is2', 't_tree', 't_sal', 't_bcf', 't_bcl',
            'f_is1', 'f_is2', 'f_tree', 'f_sal', 'f_bcf', 'f_bcp', 'f_fs',
            seg_pts_folder,
           ]
    for f in folders:
        if not make_dir(join(temp, f)): help_msg(2, f + " in the temporary directory already exists as regular file, choose another directory")

    ### Clean out some files from temporary directories that may interfere ###
    # This is needed for Mojtaba's program which simply takes folders and assumes it should process everything there
    only_keep_num(join(temp, f_d_png_folder),     zs_f, slice(-4), '*.png')
    only_keep_num(join(temp, t_d_png_folder),     zs_t, slice(-4), '*.png')
    only_keep_num(join(temp, t_s_bw_png_folder),  zs_t, slice(-4), '*.png')
    only_keep_num(join(temp, f_p_png_folder),     zs_f, slice(-14), '*_cv%d_float.png' % nrounds)
    for r in rounds:
        only_keep_num(join(temp, t_p_png_folder), zs_t, slice(-17), '*_train_round%d.txt' % r)
        only_keep_num(join(temp, t_p_png_folder), zs_t, slice(-14), '*_cv%d_float.png' % r)
        only_keep_num(join(temp, f_p_png_folder), zs_f, slice(-14), '*_cv%d_float.png' % r)
    for i in zs_t:
        only_keep_num(join(temp, t_p_png_folder), rounds, slice( 7, 8), '%04d_cv*_float.png'    % i)
        only_keep_num(join(temp, t_p_png_folder), rounds, slice(16,17), '%04d_train_round*.txt' % i)
    #for i in zs_f:
    #    f_p_png?
    only_keep_num(join(temp, t_p_png_folder), rounds, slice(15,16), 'all_train_round*.mat')
    only_keep_num(join(temp, t_p_png_folder), rounds, slice(14,15), 'MODEL_RF_round*.mat')


    ### Create the task ###
    memseg = Tasks('memseg.log',
                   {'waterlevel':wl,'pm-area-threshold-0':areaThreshold0,'pm-area-threshold-1':areaThreshold1,'pm-prob-threshold':probThreshold,
                    'contract':contract,'sigma':sigma,'number-of-trees':treeNum,'mtry':mtry,'sample-size':sampSize},
                   max_tasks_at_once = jobs, workingdir = temp,
                   rusage_log = rusage_log)
    jobs = memseg.max_tasks_at_once # if jobs was None, now it is cpu_count(), otherwise unchanged

    ### Convert input files ###
    memseg.add(('mrc2stack', mrc_f_filename, f_d_png_folder), mrc_f_filename, f_d_png).pressure(mem = 20*MB + 2*bytes_f)
    memseg.add(('mrc2stack', mrc_t_filename, t_d_png_folder), mrc_t_filename, t_d_png).pressure(mem = 20*MB + 2*bytes_t)

    memseg.add(('mrc2stack', '-emha', '-mfloat', '-s'+str(sigma), mrc_f_filename, f_d_blur_folder), mrc_f_filename, f_d_blur, 'sigma').pressure(mem = 20*MB + bytes_f + 4*pxls_f)
    memseg.add(('mrc2stack', '-emha', '-mfloat', '-s'+str(sigma), mrc_t_filename, t_d_blur_folder), mrc_t_filename, t_d_blur, 'sigma').pressure(mem = 20*MB + bytes_t + 4*pxls_t)

    memseg.add(create_inv_bw_mask_cmd(mod_t_filename, mrc_t_filename, t_s_bw,  contract), (mod_t_filename, mrc_t_filename), t_s_bw , 'contract').pressure(mem = 20*MB + bytes_t + 1*pxls_t) # TODO: support extra args
    memseg.add(create_color_mask_cmd (mod_t_filename, mrc_t_filename, t_s_clr, contract), (mod_t_filename, mrc_t_filename), t_s_clr, 'contract').pressure(mem = 20*MB + bytes_t + 3*pxls_t) # TODO: support extra args

    memseg.add(('mrc2stack',                     t_s_bw,  t_s_bw_png_folder ), t_s_bw , t_s_bw_png ).pressure(mem = 20*MB + 2*pxls_t)
    memseg.add(('mrc2stack', '-emha', '-mlabel', t_s_clr, t_s_clr_mha_folder), t_s_clr, t_s_clr_mha).pressure(mem = 20*MB + 7*pxls_t)


    ### Generate membrane segmentation from Mojtaba's code and convert resulting files ###
    #memseg.add(('moj-seg', t_d_png_folder, t_s_bw_png_folder, f_d_png_folder, f_p_png_folder, t_p_png_folder), t_d_png+t_s_bw_png+f_d_png, f_p_png+t_p_png)
    for r in rounds:
        tpr, tpr1 = t_p_png_temp[r], t_p_png_temp[r-1]
        fpr, fpr1 = f_p_png_temp[r], f_p_png_temp[r-1]
        all_train = join(t_p_png_folder, 'all_train_round%d.mat' % r)
        MODEL_RF  = join(t_p_png_folder, 'MODEL_RF_round%d.mat'  % r)
        memseg.add(('moj-seg-setup', r, t_d_png_folder, t_s_bw_png_folder, t_p_png_folder), t_d_png+t_s_bw_png+tpr1, all_train)
        memseg.add(('moj-seg-randomforest', r, t_p_png_folder), all_train, MODEL_RF).pressure(mem = 20*GB)
        [memseg.add(('moj-seg-genout', str(n+1).rjust(4, '0'), r, 'train', td, t_p_png_folder, MODEL_RF), (MODEL_RF, td),        tpr[n]).pressure(mem = 3*GB) for n, td in enumerate(t_d_png)]
        [memseg.add(('moj-seg-genout', str(n+1).rjust(4, '0'), r, 'test',  fd, f_p_png_folder, MODEL_RF), [MODEL_RF, fd] + fpr1, fpr[n]).pressure(mem = 5*GB) for n, fd in enumerate(f_d_png)]
    
    [memseg.add(('conv_img', '-mfloat', png, mha), png, mha).pressure(mem = 20*MB + 6*pxls_t) for png, mha in izip(t_p_png, t_p_mha)]
    [memseg.add(('conv_img', '-mfloat', png, mha), png, mha).pressure(mem = 20*MB + 6*pxls_f) for png, mha in izip(f_p_png, f_p_mha)]
    if sigma == 0.0:
        # TODO: does this actually work? I probably should just use a Python function to copy all at once instead of seperate processes
        [memseg.add(('cp', mha, blur), mha, blur, 'sigma').pressure(mem = 20*MB) for mha, blur in izip(t_p_mha, t_p_blur)]
        [memseg.add(('cp', mha, blur), mha, blur, 'sigma').pressure(mem = 20*MB) for mha, blur in izip(f_p_mha, f_p_blur)]
    else:
        [memseg.add(('conv_img', '-mfloat', '-s'+str(sigma), png, blur), png, blur, 'sigma').pressure(mem = 20*MB + 6*pxls_t) for png, blur in izip(t_p_png, t_p_blur)]
        [memseg.add(('conv_img', '-mfloat', '-s'+str(sigma), png, blur), png, blur, 'sigma').pressure(mem = 20*MB + 6*pxls_f) for png, blur in izip(f_p_png, f_p_blur)]


    ### Training Phase ###
    # 1 - Training texures
    memseg.add(genTextonDict_cmd(t_d_blur, t_s_clr_mha, textondict), t_d_blur+t_s_clr_mha, textondict).pressure(cpu=max(jobs*3//4, min(jobs, 2)))
    # 2 - Training initial segmentation (watershed)
    # Defaults used:
    #   [3] writeToUInt16Image       -> 0 (means write uint32 label image which is what we want)
    #   [4] keepWatershedLine        -> 1 (must be 1)
    #   [5] isBoundaryFullyConnected -> 1 (must be 1)
    [memseg.add(('hnsWatershed', pb, wl, iseg), pb, iseg, 'waterlevel') for pb, iseg in izip(t_p_blur, t_is1)]
    # 3 - Training initial segmentation (pre-merging)
    # Defaults used:
    #   [6] writeToUInt16Image       -> 0 (means write uint32 label image which is what we want)
    #   [7] consecutiveOutputLabels  -> 1 (must be 1)
    [memseg.add(('hnsMerge', iseg1, pb, areaThreshold0, areaThreshold1, probThreshold, iseg2), (iseg1, pb), iseg2, ('pm-area-threshold-0', 'pm-area-threshold-1', 'pm-prob-threshold')) for iseg1, pb, iseg2 in izip(t_is1, t_p_blur, t_is2)]
    # 4 - Training merge generation
    [memseg.add(('hnsGenMerges', iseg, pb, t, s), (iseg, pb), (t, s)) for iseg, pb, t, s in izip(t_is2, t_p_blur, t_tree, t_sal)]
    # 5 - Training boundary feature generation
    # Note that the following are now optional and can have 'NULL' instead
    #   [4] rawImageName
    #   [5] pbImageName
    #   [6] textonDictFileName (skipping this will largely accelerate the program but may worsen the accuracy by a little)
    [memseg.add(('hnsGenBoundaryFeatures', iseg, t, s, db, p, textondict, bcf), (iseg, t, s, db, p, textondict), bcf) for iseg, t, s, db, p, bcf in izip(t_is2, t_tree, t_sal, t_d_blur, t_p_mha, t_bcf)]
    # 6 - Training boundary label generation
    [memseg.add(('hnsGenBoundaryLabels', iseg, t, l, bcl), (iseg, t, l), bcl) for iseg, t, l, bcl in izip(t_is2, t_tree, t_s_clr_mha, t_bcl)]
    # 7 - Training Data Generation
    memseg.add(rf_train_cmd(t_bcf, t_bcl, treeNum, mtry, sampSize, bcmodel), t_bcf+t_bcl, bcmodel, ('number-of-trees','mtry','sample-size'))


    ### Segmentation Phase ###
    # 2 - Full dataset initial segmentation (see notes above)
    [memseg.add(('hnsWatershed', pb, wl, iseg), pb, iseg, 'waterlevel') for pb, iseg in izip(f_p_blur, f_is1)]
    # 3 - Training initial segmentation (pre-merging) (see notes above)
    [memseg.add(('hnsMerge', iseg1, pb, areaThreshold0, areaThreshold1, probThreshold, iseg2), (iseg1, pb), iseg2, ('pm-area-threshold-0', 'pm-area-threshold-1', 'pm-prob-threshold')) for iseg1, pb, iseg2 in izip(f_is1, f_p_blur, f_is2)]
    # 4 - Full dataset merge generation
    [memseg.add(('hnsGenMerges', iseg, pb, t, s), (iseg, pb), (t, s)) for iseg, pb, t, s in izip(f_is2, f_p_blur, f_tree, f_sal)]
    # 5 - Full dataset boundary feature generation (see notes above)
    [memseg.add(('hnsGenBoundaryFeatures', iseg, t, s, db, p, textondict, bcf), (iseg, t, s, db, p, textondict), bcf) for iseg, t, s, db, p, bcf in izip(f_is2, f_tree, f_sal, f_d_blur, f_p_mha, f_bcf)]
    # 8 - Generate Predictions
    rf_predict_procs(memseg, bcmodel, f_bcf, f_bcp)
    # 9 - Segment
    # Defaults used:
    #   [4] labelOutputBinaryImageConnectedComponents -> 1 (must be 1)
    #   [5] writeToUInt16Image       -> 0 (means write uint32 label image which is what we want)
    [memseg.add(('hnsSegment', iseg, t, bcp, fseg), (iseg, t, bcp), fseg) for iseg, t, bcp, fseg in izip(f_is2, f_tree, f_bcp, f_fs)]
    
    ### Convert output files ###
    [memseg.add(('hnsGenOrderedContours', fseg, z, sp), fseg, sp) for fseg, z, sp in izip(f_fs, zs_f, seg_pts)]
    memseg.add(['combine_points',] + seg_pts + [seg_pts_all,], seg_pts, seg_pts_all)
    # TODO: -im and pixel spacing?
    memseg.add(('point2model', '-im', mrc_f_filename, seg_pts_all, mod_output), (mrc_f_filename,seg_pts_all), mod_output)


    # Run!
    memseg.run(verbose = True)
