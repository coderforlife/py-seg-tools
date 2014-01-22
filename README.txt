=== Python MRC Image Conversion Tools ===

These scritps require NumPy, SciPy, and PIL (python-imaging). An optional
module is h5py which is required for reading v7.3 MATLAB images.

On Linux you can install them using something similar to:
 $ sudo yum install numpy scipy python-imaging libhdf5-dev
 $ sudo easy_install h5py
On Windows I recommend installing Python(x,y) which includes everything.

Currently there are three Python scripts:
 * mrc2stack
 * stack2mrc
 * conv_img

--- Formats Supported ---
For images, any format supported by SciPy / PIL is supported in addition to
MetaFile formats (MHA and MHD) and reading MAT files. MHA/MHD/MAT files
normally support many images per file, but only the first is read.

Common supported SciPy / PIL formats:
 * PNG  (1-bit BW, 8-bit gray, 24-bit RGB)
 * BMP  (1-bit BW, 8-bit gray, 24-bit RGB)
 * TIFF (1-bit BW, 8-bit gray, 24-bit RGB)
 * JPEG (8-bit gray, 24-bit RGB)
See http://www.pythonware.com/library/pil/handbook/formats.htm for all details

The format used is based on the file extension (included in the output for
conv_img or the ext param in mrc2stack). See below for more information.

Note: in all cases the output file format must support either the same pixel
format as the input data or the converted mode. For example, PNG does not
support floating-point formats so using -mfloat (see below) will not work and
writing to a PNG will not work.


--- Image Processing Options ---
All programs support the options --sigma (or -s) and --flip (or -f). Sigma
blurs the image with a Gaussian blur using the given sigma value (positive real
number). Flipping flips the image top to bottom when saving.

For mrc2stack and conv_img you can also convert the data itself by using --mode
(or -m) with one of the following:
 * 'float'   - output a 32-bit floating-point number output scaled to 0.0-1.0
 * 'label'   - output an consecutively numbered image using con. components
 * 'relabel' - output an consecutively re-numbered label data image


--- Conversion from MRC stack: mrc2stack ---
MRC files are any files that are IMOD image stacks (they may have different
file extensions, such as REC, ST, ALI, or PRE-ALI). The input is the MRC file
while the output is a directory where all the different slices will be output.
Currently MRC files in complex format (modes 3 and 4) are unsupported.

You can use -x #-# and -y #-# to extract a subimage for each section.
Additionally  you can use -z to specify which slices to extract. "z" supports
any combination of comma seperated slices or ranges using a dash (e.g. 3,5-9).

The default output files are ###.png where ### is a 3-digit number from the
slice it came from, with leading 0s if necessary. To change this use -e or
--ext to change the file extension and -b or --base to change the basename,
using %03d for the slice number as 3 digits with leading zeros or just %d for
the number.


--- Conversion to MRC stack: stack2mrc ---
When writing an MRC you supply many input files before you give the output MRC
file. The input files can be different file types but must all have the same
pixel format and dimensions. Additionally, you can use inputs that have
glob-like syntax (e.g. folder/*.png, [0-9][0-9].png, etc).


== Python Coding ==
You can also directly call the above tools from Python code by importing the
function from the file (e.g. from mrc2stack import mrc2stack). Use the built-in
Python help functions for the documentation of each function.

There is also a complete MRC class and various other utility functions in
images.py. MRC class is currently not really documented.

The utility functions (for more details see built-in Python help):
	gauss_blur(im, sigma = 1.0) -- Blur an image using a Gaussian blur (requires SciPy)
	flip_up_down(im)            -- Flips an image from top-bottom as a view (not a copy)
	label(im)                   -- Creates a consecutively numbered image using connected components
	relabel(im)                 -- Renumbers an image to be consecutively numbered
	float_image(im, in_scale=None, out_scale=(0.0,1.0)) - Convert an image into a 32-bit floating-point image by scaling the data
	imread(filename)            -- Read an image
	imsave(filename, im)        -- Save an image
	is_rgb24(im)                -- True if the data represents an 24-bit RGB image
	is_image_besides_rgb24(im)  -- True if the data represents an image beside a 24-bit RGB image
	is_image(im)                -- True if the data represents an image

Additionally, there are dtypes for the different image types:
	IM_BYTE      -- unsigned 8-bit grayscale
	IM_SBYTE     -- signed 8-bit grayscale
	IM_SHORT     -- signed 16-bit grayscale (little-endian)
	IM_SHORT_BE  -- signed 16-bit grayscale (big-endian)
	IM_USHORT    -- unsigned 16-bit grayscale (little-endian)
	IM_USHORT_BE -- unsigned 16-bit grayscale (big-endian)
	IM_INT       -- signed 32-bit grayscale (little-endian)
	IM_INT_BE    -- signed 32-bit grayscale (big-endian)
	IM_UINT      -- unsigned 32-bit grayscale (little-endian)
	IM_UINT_BE   -- unsigned 32-bit grayscale (big-endian)
	IM_LONG      -- signed 64-bit grayscale (little-endian)
	IM_LONG_BE   -- signed 64-bit grayscale (big-endian)
	IM_ULONG     -- unsigned 64-bit grayscale (little-endian)
	IM_ULONG_BE  -- unsigned 64-bit grayscale (big-endian)
	IM_RGB24     -- 24-bit RGB (color is on third axis as bytes, cannot be easily tested)
	IM_RGB24_STRUCT -- 24-bit RGB (color is with R, G, and B fields as bytes)
	IM_FLOAT     -- 32-bit floating-point grayscale
	IM_DOUBLE    -- 64-bit floating-point grayscale
