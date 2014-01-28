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

import os
from sys import stdout, stderr, argv, exit
if hasattr(stdout, 'fileno'):
    try:
        stdout = os.fdopen(stdout.fileno(), 'w', 0)
        stderr = os.fdopen(stderr.fileno(), 'w', 0)
    except: pass


def imodmop_cmd(args, model, in_mrc, out_mrc, contract = 0):
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
    tw = TextWrapper(width = w, subsequent_indent = ' '*21)
    if msg != None: print >> stderr, fill(msg, w)
    print "Usage:"
    print tw.fill("  %s [args] training.mrc training.mod full.mrc output.mod" % basename(argv[0]))
    print ""
    print "Optional algorithm parameters:"
    print tw.fill("  -w  --water-lvl=   The watershed water level parameter, probably <=0.02, if not provided will calculate an acceptable value and save it in the temp directory")
    print tw.fill("  -c  --contract=    The amount to contract contours by to make them inside the membranes")
    print tw.fill("  -s  --sigma=       The amount of Gaussian blur to use, default is 1.0 while 0.0 turns off blurring")
    print tw.fill("  -S  --chm-nstage=  The number of stages of processing to perform during CHM segmentation, default is 2")
    print tw.fill("  -L  --chm-nlevel=  The number of levels of processing to perform during each stage of CHM segmentation, default is 4")
    print tw.fill("  -O  --chm-overlap= The overlap of the blocks used when CHM testing specified as a single number or two numbers (for X and Y), default is 25x25")
    print tw.fill("  --no-histeq        Do not perform histogram equalization on testing data using the histogram from the training data");
    print tw.fill("  --num-trees=       Number of random forest trees, should be at least 100, default is 255 - larger will be slower")
    print tw.fill("  --mtry=            Number of features to use in each node in RF, should be <<85, default is sqrt(total features)")
    print tw.fill("  --samp-size=       Fraction of samples used in each node in RF, default is 0.7")
    print tw.fill("  --pm-area-thresh0= Pre-merge area threshold #1, default is 50")
    print tw.fill("  --pm-area-thresh1= Pre-merge area threshold #2, default is 200")
    print tw.fill("  --pm-prop-thresh=  Pre-merge average probability threshold, default is 0.5")
    print ""
    print "Other optional arguments:"
    print tw.fill("  -h  --help         Display this help")
    print tw.fill("  -t  --temp=        Set temporary directory, default value is ./temp")
    print tw.fill("  -j  --jobs=        Maximum number of jobs to do at once, default is num of processors")
    print tw.fill("  -u  --rusage=      Save the resources usage (memory and time) for each run process to a file")
    print tw.fill("  -C  --cluster=     Use the cluster specified by the file for some operations, defaults to running everything locally")
    exit(err)

def die(err, msg):
    print >> stderr, msg
    exit(err)



inf = float("inf")
def __get_float_err(name, range):
    lo_inf, hi_inf = range[0] == -inf, range[1] == inf
    if lo_inf and hi_inf: return "'%s' must be a floating-point number" % (name)
    elif lo_inf: return "'%s' must be a floating-point number less than or equal to %f" % (name, range[1])
    elif hi_inf: return "'%s' must be a floating-point number greater than or equal to %f" % (name, range[0])
    return "'%s' must be a floating-point number between %f and %f" % (name, range[0], range[1])
    
def get_float(o, a, options, name, old_value, ref, range=(-inf, inf)): # range is inclusive
    if o not in options: return False
    if old_value != None: help_msg(2, "Must be only one '%s' argument" % name)
    try: x = float(a)
    except: help_msg(2, __get_float_err(name,range))
    if isnan(x) or x < range[0] or x > range[1]: help_msg(2, __get_float_err(name,range))
    ref[0] = x
    return True
def get_int_pos(o, a, options, name, old_value, ref, max_val = -1):
    if o not in options: return False
    if old_value != None: help_msg(2, "Must be only one '%s' argument" % name)
    if not a.isdigit() or int(a) <= 0: help_msg(2, "'%s' must be a positive integer" % name)
    if max_val > 0 and int(a) > max_value: help_msg(2, "'%s' must be a positive integer less than or equal to %d" % (name, max_value))
    ref[0] = int(a)
    return True


if __name__ == "__main__":
    from getopt import getopt, error as getopt_error
    from glob import glob
    from os.path import exists, isdir, join, realpath, relpath
    from os import getcwd
    from math import isnan
    from itertools import izip, product

    #argv = [argv[0], '-w0.01', 'training.mrc', 'training.mod', 'full.mrc', 'output.mod']

    if len(argv) < 2: help_msg(1)

    try:
        opts, args = getopt(argv[1:], "ht:j:u:C:w:c:s:S:L:O:",
                            ["help", "temp=", "jobs=", "rusage=", "cluster=",
                             "no-histeq", "water-lvl=", "contract=", "sigma=", "chm-nstage=", "chm-nlevel=", "chm-overlap=",
                             "num-trees=", "mtry=", "samp-size=", "pm-area-thresh0=", "pm-area-thresh1=", "pm-prop-thresh="
                             ])
    except getopt_error, msg: help_msg(2, msg)

    # Check the arguments
    temp = None
    jobs = None
    rusage_log = None
    cluster = None
    histeq = True
    wl = None
    contract = None
    sigma = None
    chm_nstage = None
    chm_nlevel = None
    chm_overlap = None
    treeNum = None
    mtry = None
    sampSize = None
    areaThreshold0 = None
    areaThreshold1 = None
    probThreshold = None
    ref = [0]
    for o,a in opts:
        if o == "-h" or o == "--help":    help_msg()
        elif o == "-t" or o == "--temp":
            if temp != None: help_msg(2, "Must be only one 'temp' argument")
            temp = realpath(a)
        elif get_int_pos(o, a, ("-j", "--jobs="), "jobs", jobs, ref): jobs = ref[0]
        elif o == "-u" or o == "--rusage":
            if rusage_log != None: help_msg(2, "Must be only one 'rusage' argument")
            try:
                from os_ext import wait4 # make sure wait4 is available
                rusage_log = realpath(a)
            except ImportError:
                print >> stderr, "Warning: System does not support recording resource usage, 'rusage' argument ignored."
        elif o == "-C" or o == "--cluster":
            if cluster != None: help_msg(2, "Must be only one 'cluster' argument")
            try: from cluster import Cluster
            except ImportError: print >> stderr, "Warning: Cluster service requires the SAGA Python module. See saga-project.github.io. Cluster will not be used."
            try: cluster = Cluster(a)
            except Exception:   print >> stderr, "Warning: Cluster information could not be read. Cluster will not be used."
        elif o == "no-histeq":
            if not histeq: help_msg(2, "Must be only one 'no-histeq' argument")
            histeq = False
        elif get_float  (o, a, ("-w","--water-lvl"  ), "water-lvl",       wl,             ref, (0.0, 1.0)): wl             = ref[0]
        elif get_float  (o, a, ("-c","--contract"   ), "contract",        contract,       ref            ): contract       = ref[0]
        elif get_float  (o, a, ("-s","--sigma"      ), "sigma",           sigma,          ref, (0.0, inf)): sigma          = ref[0]
        elif get_int_pos(o, a, ("-S","--chm-nstage" ), "chm-nstage",      chm_nstage,     ref            ): chm_nstage     = ref[0]
        elif get_int_pos(o, a, ("-L","--chm-nlevel" ), "chm-nlevel",      chm_nlevel,     ref            ): chm_nlevel     = ref[0]
        elif o == "-O" or o == "--chm-overlap":
            if chm_overlap != None: die("Must be only one 'chm_overlap' argument", 2)
            parts = a.split('x')
            if len(parts) > 2 or not all(x.isdigit() for x in parts): help_msg(2, "The 'chm_overlap' argument must be either a single non-negative integer or two in the form #x#")
            chm_overlap = (int(parts[0]), int(parts[len(parts) - 1]))
            if chm_overlap[0] < 0 or chm_overlap[1] < 0: help_msg(2, "The 'chm_overlap' argument must be either a single non-negative integer or two in the form #x#")
        elif get_int_pos(o, a, ("--num-trees",      ), "num-trees",       treeNum,        ref            ): treeNum        = ref[0]
        elif get_int_pos(o, a, ("--mtry",           ), "mtry",            mtry,           ref, 40        ): treeNum        = ref[0]
        elif get_float  (o, a, ("--samp-size",      ), "samp-size",       sampSize,       ref, (0.0, 1.0)): sampSize       = ref[0]
        elif get_int_pos(o, a, ("--pm-area-thresh0",), "pm-area-thresh0", areaThreshold0, ref            ): areaThreshold0 = ref[0]
        elif get_int_pos(o, a, ("--pm-area-thresh1",), "pm-area-thresh1", areaThreshold1, ref            ): areaThreshold1 = ref[0]
        elif get_float  (o, a, ("--pm-prop-thresh" ,), "pm-prop-thresh",  probThreshold,  ref, (0.0, 1.0)): probThreshold  = ref[0]

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
    #if wl          == None: # None means calculate best-guess
    #if jobs        != None: ... # dealt with later
    #if rusage_log  == None: # None means no log
    #if cluster     == None: # None means no custer used
    if contract    == None: contract = 0
    if sigma       == None: sigma = 1.0
    if chm_nstage  == None: chm_nstage = 2
    if chm_nlevel  == None: chm_nlevel = 4
    if chm_overlap == None: chm_overlap = (25, 25)
    if treeNum     == None: treeNum = 255
    if mtry        == None: mtry = 0 # will make the program calculate the actual proper default
    if sampSize    == None: sampSize = 0.70
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
    pxls_t  = mrc_t.section_size
    bytes_t = mrc_t.section_full_data_size
    pxls_f  = mrc_f.section_size
    bytes_f = mrc_f.section_full_data_size
    chm_block_size = '%dx%d' % (mrc_t.nx, mrc_t.ny)
    chm_overlap    = '%dx%d' % chm_overlap
    zs_t = range(len(mrc_t))
    zs_f = range(len(mrc_f))
    mrc_t.close()
    mrc_f.close()

    # Generic filenames
    tifs_t = ['%04d.tif' % i for i in zs_t]
    tifs_f = ['%04d.tif' % i for i in zs_f]
    mhas_t = ['%04d.mha' % i for i in zs_t]
    mhas_f = ['%04d.mha' % i for i in zs_f]
    ssvs_t = ['%04d.ssv' % i for i in zs_t]
    ssvs_f = ['%04d.ssv' % i for i in zs_f]
    t_chm_files = '####.tif;%d-%d' % (zs_t[0], zs_t[-1])
    f_chm_files = '####.tif;%d-%d' % (zs_f[0], zs_f[-1])


    # Notes on my conventions:
    # Since there are so many variables coming up, I starting a naming system for them
    # They begin with "f_" and "t_" to represent the full and training datasets respectively
    # The first set (during image conversions) have "d_" for raw data, "s_" for label/segmented data, and "p_" for probability maps
    # The end is the file type (TIFF, MHA, or MHA-blurred [simply blur])
    # Other names end with some sort of short descriptor of the contents (is = initial segmentation, tree = seg tree, sal = seg saliency, ...)

    ## All of the file names that will be used, relative to temporary directory
    if histeq: histogram = 'histogram.txt'
    t_d_tif_folder = 't_d_tif'
    f_d_tif_folder = 'f_d_tif'
    t_d_blur_folder = 't_d_blur'
    f_d_blur_folder = 'f_d_blur'
    t_d_tif    = [join(t_d_tif_folder,  i) for i in tifs_t] # training dataset (TIFF)
    f_d_tif    = [join(f_d_tif_folder,  i) for i in tifs_f] # full dataset     (TIFF)
    t_d_blur   = [join(t_d_blur_folder, i) for i in mhas_t] # training dataset (MHA-blurred)
    f_d_blur   = [join(f_d_blur_folder, i) for i in mhas_f] # full dataset     (MHA-blurred)

    t_s_bw     = 't_s_bw.mrc'  # training labels (black and white)
    t_s_clr    = 't_s_clr.mrc' # training labels (colored)
    t_s_bw_tif_folder = 't_s_bw'
    t_s_clr_mha_folder = 't_s_clr'
    t_s_bw_tif = [join(t_s_bw_tif_folder,  i) for i in tifs_t] # training labels (TIFF-black and white)
    t_s_clr_mha= [join(t_s_clr_mha_folder, i) for i in mhas_t] # training labels (MHA-colored)

    chm_working_folder = 'chm_temp'
    chm_model_files = ([join(chm_working_folder, 'param.mat'), join(chm_working_folder, 'MODEL_level0_stage%d.mat' % chm_nstage)] +
                       [join(chm_working_folder, 'MODEL_level%d_stage%d.mat' % (l,s)) for s, l in product(xrange(1,chm_nstage), xrange(chm_nlevel+1))])
    
    t_p_mat_folder = join(chm_working_folder, 'output_level0_stage%d' % chm_nstage)
    t_p_blur_folder = 't_p_blur'
    t_p_mat    = [join(t_p_mat_folder, '%04d.mat' % i) for i in zs_t] # training probabilty map (MAT)
    t_p_mha    = [join('t_p_mha',       i) for i in mhas_t] # training probabilty map (MHA)
    t_p_blur   = [join(t_p_blur_folder, i) for i in mhas_t] # training probabilty map (MHA-blurred)

    f_p_tif_folder = 'f_p_tif'
    f_p_tif    = [join(f_p_tif_folder,  i) for i in tifs_f] # full probabilty map (TIFF)
    f_p_mha    = [join('f_p_mha',       i) for i in mhas_f] # full probabilty map (MHA)
    f_p_blur   = [join('f_p_blur',      i) for i in mhas_f] # full probabilty map (MHA-blurred)

    textondict             = 'textondict.ssv' # Texture data
    if wl == None: wl_file = 'waterlevel.txt' # Waterlevel value file

    t_is1      = [join('t_is1',  i) for i in mhas_t] # training initial segmentation (from watershed)
    t_is2      = [join('t_is2',  i) for i in mhas_t] # training initial segmentation (from pre-merging)
    t_tree     = [join('t_tree', i) for i in ssvs_t] # training segmentation tree
    t_sal      = [join('t_sal',  i) for i in ssvs_t] # training segmentation saliency
    t_bcf      = [join('t_bcf',  i) for i in ssvs_t] # training segmentation features
    t_bcl      = [join('t_bcl',  i) for i in ssvs_t] # training segmentation labels

    bcmodel = 'bcmodel' # Training data

    f_is1      = [join('f_is1',  i) for i in mhas_f] # full initial segmentation (from watershed)
    f_is2      = [join('f_is2',  i) for i in mhas_f] # full initial segmentation (from pre-merging)
    f_tree     = [join('f_tree', i) for i in ssvs_f] # full segmentation tree
    f_sal      = [join('f_sal',  i) for i in ssvs_f] # full segmentation saliency
    f_bcf      = [join('f_bcf',  i) for i in ssvs_f] # full segmentation features
    f_bcp      = [join('f_bcp',  i) for i in ssvs_f] # full segmentation predictions
    f_fs       = [join('f_fs',   i) for i in mhas_f] # the final segementation

    seg_pts_folder = 'seg_pts'
    seg_pts    = [join(seg_pts_folder,'%04d.pts' % i) for i in zs_f] # the final segementation points for each section
    seg_pts_all = 'segmentation.pts' # the final segementation points for all sections

    # All folders that are used
    folders = [
            t_d_tif_folder, f_d_tif_folder, t_d_blur_folder, f_d_blur_folder, t_s_bw_tif_folder, t_s_clr_mha_folder,
            chm_working_folder,
            t_p_mat_folder, 't_p_mha', 't_p_blur',
            f_p_tif_folder, 'f_p_mha', 'f_p_blur',
            't_is1', 't_is2', 't_tree', 't_sal', 't_bcf', 't_bcl',
            'f_is1', 'f_is2', 'f_tree', 'f_sal', 'f_bcf', 'f_bcp', 'f_fs',
            seg_pts_folder,
           ]
    for f in folders:
        if not make_dir(join(temp, f)): help_msg(2, f + " in the temporary directory already exists as regular file, choose another directory")


    ### Create the task ###
    memseg = Tasks('memseg.log',
                   {'contract':contract,'sigma':sigma,'histeq':histeq,
                    'chm-nstage':chm_nstage,'chm-nlevel':chm_nlevel,'chm-overlap':chm_overlap,
                    'waterlevel':wl,'pm-area-threshold-0':areaThreshold0,'pm-area-threshold-1':areaThreshold1,'pm-prob-threshold':probThreshold,
                    'number-of-trees':treeNum,'mtry':mtry,'sample-size':sampSize},
                   max_tasks_at_once = jobs, workingdir = temp,
                   rusage_log = rusage_log)
    jobs = memseg.max_tasks_at_once # if jobs was None, now it is cpu_count(), otherwise unchanged
    most_jobs = max(jobs*3//4, min(jobs, 2))
    least_jobs = max(jobs - most_jobs, 1)
    blur = '-s'+str(sigma)
    hgram = '-h'+histogram

    ### Convert input files ###
    # TODO: Decide what should be done with background
    # Testing data only, or training too?
    # f_d_tif have background removed or mirrored? (and if mirrored, removed or set to black after CHM testing?)
    # f_d_blur have background removed or mirrored?
    
    memseg.add(('mrc2stack', '-etif', mrc_t_filename, t_d_tif_folder), mrc_t_filename, t_d_tif).pressure(mem = 20*MB + 2*bytes_t)
    memseg.add(('mrc2stack', '-emha', blur, '-F', mrc_t_filename, t_d_blur_folder), mrc_t_filename, t_d_blur, 'sigma').pressure(mem = 20*MB + bytes_t + 4*pxls_t)

    if histeq:
        memseg.add(['get_histogram'] + t_d_tif + [histogram], t_d_tif, histogram, 'histeq').pressure(mem = 20*MB + 2*bytes_t)
        memseg.add(('mrc2stack', '-etif', hgram,             mrc_f_filename, f_d_tif_folder), (mrc_f_filename, histogram), f_d_tif, 'histeq').pressure(mem = 20*MB + 2*bytes_f)
        memseg.add(('mrc2stack', '-emha', hgram, blur, '-F', mrc_f_filename, f_d_blur_folder), mrc_f_filename, f_d_blur, ('sigma','histeq')).pressure(mem = 20*MB + bytes_f + 4*pxls_f)
    else:
        memseg.add(('mrc2stack', '-etif',             mrc_f_filename, f_d_tif_folder), mrc_f_filename, f_d_tif, 'histeq').pressure(mem = 20*MB + 2*bytes_f)
        memseg.add(('mrc2stack', '-emha', blur, '-F', mrc_f_filename, f_d_blur_folder), mrc_f_filename, f_d_blur, ('sigma','histeq')).pressure(mem = 20*MB + bytes_f + 4*pxls_f)

    memseg.add(create_inv_bw_mask_cmd(mod_t_filename, mrc_t_filename, t_s_bw,  contract), (mod_t_filename, mrc_t_filename), t_s_bw , 'contract').pressure(mem = 20*MB + bytes_t + 1*pxls_t)
    memseg.add(create_color_mask_cmd (mod_t_filename, mrc_t_filename, t_s_clr, contract), (mod_t_filename, mrc_t_filename), t_s_clr, 'contract').pressure(mem = 20*MB + bytes_t + 3*pxls_t)

    memseg.add(('mrc2stack', '-etif',       t_s_bw,  t_s_bw_tif_folder ), t_s_bw , t_s_bw_tif ).pressure(mem = 20*MB + 2*pxls_t)
    memseg.add(('mrc2stack', '-emha', '-R', t_s_clr, t_s_clr_mha_folder), t_s_clr, t_s_clr_mha).pressure(mem = 20*MB + 7*pxls_t)


    ### Generate membrane segmentation from Mojtaba's code and convert resulting files ###
    #memseg.add(('CHMSEG', join(t_d_tif_folder, t_chm_files), join(t_s_bw_tif_folder, t_chm_files), join(f_d_tif_folder, f_chm_files), f_p_tif_folder, chm_working_folder), t_d_tif+t_s_bw_tif+f_d_tif, f_p_tif+t_p_mat)
    memseg.add(('CHM_train', join(t_d_tif_folder, t_chm_files), join(t_s_bw_tif_folder, t_chm_files), chm_working_folder, chm_nstage, chm_nlevel), t_d_tif+t_s_bw_tif, t_p_mat+chm_model_files, 'chm-nstage', 'chm-nlevel').pressure(mem=75*GB, cpu=least_jobs)
    [memseg.add(('CHM_test', fd, f_p_tif_folder, '-s', '-m', chm_working_folder, '-b', chm_block_size, '-o', chm_overlap), [fd] + chm_model_files, fp, 'chm-overlap', can_run_on_cluster=True).pressure(mem=10*GB) for fd, fp in zip(f_d_tif, f_p_tif)]
    
    [memseg.add(('conv_img',       mat, mha), mat, mha).pressure(mem = 20*MB + 6*pxls_t) for mat, mha in izip(t_p_mat, t_p_mha)]
    [memseg.add(('conv_img', '-F', tif, mha), tif, mha).pressure(mem = 20*MB + 6*pxls_f) for tif, mha in izip(f_p_tif, f_p_mha)]
    if sigma == 0.0:
        # TODO: does this actually work? I probably should just copy all at once instead of seperate processes
        [memseg.add(('cp', mha, blur), mha, blur, 'sigma').pressure(mem = 20*MB) for mha, blur in izip(t_p_mha, t_p_blur)]
        [memseg.add(('cp', mha, blur), mha, blur, 'sigma').pressure(mem = 20*MB) for mha, blur in izip(f_p_mha, f_p_blur)]
    else:
        [memseg.add(('conv_img', blur,       mat, blur), mat, blur, 'sigma').pressure(mem = 20*MB + 6*pxls_t) for mat, blur in izip(t_p_mat, t_p_blur)]
        [memseg.add(('conv_img', blur, '-F', tif, blur), tif, blur, 'sigma').pressure(mem = 20*MB + 6*pxls_f) for tif, blur in izip(f_p_tif, f_p_blur)]


    ### Training Phase ###
    # 1 - Training texures
    memseg.add(genTextonDict_cmd(t_d_blur, t_s_clr_mha, textondict), t_d_blur+t_s_clr_mha, textondict).pressure(cpu=most_jobs)
    # 2 - Training initial segmentation (watershed)
    # Defaults used:
    #   [3] writeToUInt16Image       -> 0 (means write uint32 label image which is what we want)
    #   [4] keepWatershedLine        -> 1 (must be 1)
    #   [5] isBoundaryFullyConnected -> 1 (must be 1)
    if wl == None:
        ## TODO: create hnsCalculateWatershedThreshold
        memseg.add(('hnsCalculateWatershedThreshold', zs_t[0], zs_t[-1], join(t_p_blur_folder, '%04d.mha'), join(t_s_clr_mha_folder, '%04d.mha'), areaThreshold0, areaThreshold1, probThreshold),
                   (t_p_blur, t_s_clr_mha), wl_file, ('waterlevel', 'pm-area-threshold-0', 'pm-area-threshold-1', 'pm-prob-threshold'), stdout=wl_file).pressure(cpu=4)
    [memseg.add(('hnsWatershed', pb, wl if wl else wl_file, iseg), pb if wl else (pb, wl_file), iseg, 'waterlevel') for pb, iseg in izip(f_p_blur, f_is1)]
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
    [memseg.add(('hnsGenBoundaryFeatures', iseg, t, s, db, p, textondict, bcf), (iseg, t, s, db, p, textondict), bcf, can_run_on_cluster=True) for iseg, t, s, db, p, bcf in izip(t_is2, t_tree, t_sal, t_d_blur, t_p_mha, t_bcf)]
    # 6 - Training boundary label generation
    [memseg.add(('hnsGenBoundaryLabels', iseg, t, l, bcl), (iseg, t, l), bcl) for iseg, t, l, bcl in izip(t_is2, t_tree, t_s_clr_mha, t_bcl)]
    # 7 - Training Data Generation
    memseg.add(rf_train_cmd(t_bcf, t_bcl, treeNum, mtry, sampSize, bcmodel), t_bcf+t_bcl, bcmodel, ('number-of-trees','mtry','sample-size'))


    ### Segmentation Phase ###
    # 2 - Full dataset initial segmentation (see notes above)
    [memseg.add(('hnsWatershed', pb, wl if wl else wl_file, iseg), pb if wl else (pb, wl_file), iseg, 'waterlevel') for pb, iseg in izip(f_p_blur, f_is1)]
    # 3 - Training initial segmentation (pre-merging) (see notes above)
    [memseg.add(('hnsMerge', iseg1, pb, areaThreshold0, areaThreshold1, probThreshold, iseg2), (iseg1, pb), iseg2, ('pm-area-threshold-0', 'pm-area-threshold-1', 'pm-prob-threshold')) for iseg1, pb, iseg2 in izip(f_is1, f_p_blur, f_is2)]
    # 4 - Full dataset merge generation
    [memseg.add(('hnsGenMerges', iseg, pb, t, s), (iseg, pb), (t, s)) for iseg, pb, t, s in izip(f_is2, f_p_blur, f_tree, f_sal)]
    # 5 - Full dataset boundary feature generation (see notes above)
    [memseg.add(('hnsGenBoundaryFeatures', iseg, t, s, db, p, textondict, bcf), (iseg, t, s, db, p, textondict), bcf, can_run_on_cluster=True) for iseg, t, s, db, p, bcf in izip(f_is2, f_tree, f_sal, f_d_blur, f_p_mha, f_bcf)]
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
    memseg.run(cluster=cluster, verbose=True)

    # Cleanup
    if cluster: cluster.close()
