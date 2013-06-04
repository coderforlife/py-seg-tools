=== Python MRC Image Conversion Tools ===

These scritps require NumPy, SciPy, PIL (python-imaging) and SimpleITK.
On Linux you can install them using something similar to:
 $ sudo yum install numpy scipy python-imaging
 $ sudo easy_install SimpleITK
On Windows I recommend using isntalling Python(x,y) then installing
SimpleITK from an administrator command line with "easy_install SimpleITK".

Currently there are three Python scripts:
 * mrc2png
 * mrc2mha
 * png2mha

These are a bit misnamed becaumse they actually support many image formats.
The PNG programs support any SciPy / PIL format while the MHA programs support
anyITK format.

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

When outputing to an ITK format you can perform additional conversion and
filtering. Using -s or --sigma with a positive floating point number will use a
Gaussian blur on the image with the given sigma. Using -m or --mode will
convert the image according to the mode type:
 * 'float' - output a 32-bit floating-point number output scaled to 0.0-1.0
 * 'label' - output a 32-bit unsigned consecutively numbered label data image

Note: in all cases the output file format must support either the same type as
the input data or the converted input type. For example, PNG does not support
floating-point formats so using -mfloat will not work and writing to a PNG will
not work.


--- Conversion from MRC stack: mrc2png and mrc2mha ---
MRC files are any files that are IMOD image stacks (they may have different
file extensions, such as REC, ST, ALI, or PRE-ALI). The input is the MRC file
while the output is a directory where all the different slices will be output.

You can use -i or --indices to specify which slices to extract from the MRC
file. These can use any combination of comma seperated slices or ranges using a
dash (e.g. 1,3,5-9).

The default output files are ###.png or ###.mha where ### is a 3-digit number
from the slice it came from, with leading 0s if necessary. To change this use
-b or --base to change the basename for files, using %03d for the slice number
as 3 digits with leading zeros or just %d for the number. The basename must
include the file extension. By using an extension besides png or mha the file
format will be different.

Additionally when converting to MHA you can perform some additional conversion
and filtering. Using -s or --sigma with a positive floating point number will
use a Gaussian blur on the image with the given sigma. Using -m or --mode will
convert the image according to the mode type:
 * 'float' - output a 32-bit floating-point number output scaled to 0.0-1.0
 * 'label' - output a 32-bit unsigned consecutively numbered label data image


--- Conversion between single images: png2mha ---
This converts between any SciPy / PIL supported format to any ITK supported
format. This takes a single image as input and a single image to output to.
The file extension needs to be included on the output file.

Like mrc2mha, this supports additional conversion and filtering. See the last
paragraph in the "Conversersion from MRC stack" for more information.

