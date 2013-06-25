=== Python MRC Image Conversion Tools ===

These scritps require NumPy, SciPy, PIL (python-imaging) and SimpleITK.
On Linux you can install them using something similar to:
 $ sudo yum install numpy scipy python-imaging
 $ sudo easy_install SimpleITK
On Windows I recommend installing Python(x,y) then installing SimpleITK from an
administrator command line with "easy_install SimpleITK".

Currently there are five Python scripts:
 * mrc2png
 * mrc2mha
 * png2mha
 * png2mrc
 * mha2mrc

--- Formats Supported ---
These are a bit misnamed becaumse they actually support many image formats.
The "png" means any format supported by SciPy / PIL while "mha" means any
format supported by ITK.

Common supported SciPy / PIL formats:   (faster than ITK)
 * PNG  (1-bit BW, 8-bit gray, 24-bit RGB)
 * BMP  (1-bit BW, 8-bit gray, 24-bit RGB)
 * TIFF (1-bit BW, 8-bit gray, 24-bit RGB)
 * JPEG (8-bit gray, 24-bit RGB)
See http://www.pythonware.com/library/pil/handbook/formats.htm for all details

Common supported ITK formats:   (* means advantage over SciPy/PIL)
 * *MHA/MHD (all)*
 * *VTK     (all)*
 * PNG  (8-bit gray, *16-bit gray*, 24-bit RGB)
 * BMP  (8-bit gray, 24-bit RGB)
 * TIFF (8-bit gray, *16-bit gray*, 24-bit RGB)
 * JPEG (8-bit gray)
See http://www.paraview.org/Wiki/ITK/File_Formats for a full list

The format used is based on the file extension (included in the output for
png2mha or the basename in mrc2png and mrc2mha). See below for more
information.

Note: in all cases the output file format must support either the same type as
the input data or the converted mode. For example, PNG does not support
floating-point formats so using -mfloat (see below) will not work and writing
to a PNG will not work.


--- Image Processing Options ---
All programs support the options --sigma (or -s) and --flip (or -f). Sigma
blurs the image with a Gaussian blur using the given sigma value (positive real
number). Flipping flips the image top to bottom when saving.

When outputing to an ITK format you can also convert the data itself by using
--mode (or -m) with one of the following:
 * 'float' - output a 32-bit floating-point number output scaled to 0.0-1.0
 * 'label' - output a 32-bit unsigned consecutively numbered label data image


--- Conversion from MRC stack: mrc2png and mrc2mha ---
MRC files are any files that are IMOD image stacks (they may have different
file extensions, such as REC, ST, ALI, or PRE-ALI). The input is the MRC file
while the output is a directory where all the different slices will be output.
Currently MRC files in complex format (modes 3 and 4) are unsupported.

You can use -x #-# and -y #-# to extract a subimage for each section.
Additionally  you can use -z to specify which slices to extract. "z" supports
any combination of comma seperated slices or ranges using a dash (e.g. 3,5-9).

The default output files are ###.png or ###.mha where ### is a 3-digit number
from the slice it came from, with leading 0s if necessary. To change this use
-b or --base to change the basename for files, using %03d for the slice number
as 3 digits with leading zeros or just %d for the number. The basename must
include the file extension. By using an extension besides png or mha the file
format will be different.


--- Conversion to MRC stack: png2mrc and mha2png ---
When writing an MRC you supply many input files before you give the output MRC
file. The input files can be different file types (as long as they are all
supported by the current reader: SciPy/PIL or ITK). Additionally, you can use
inputs that have glob-like syntax (e.g. folder/*.png, [0-9][0-9].png, etc).

