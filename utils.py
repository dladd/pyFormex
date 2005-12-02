#!/usr/bin/env python
# $Id$

def imageFormatFromExt(ext):
    """Determine the image format from an extension.

    The extension can start with a '.' or not and be in upper or
    lower case. The format is usually equal to the extension characters
    in upper case, except that .jpg/.JPG file extensions return a format JPEG.
    If the supplied extension is empty, the default format 'PNG' is returned.
    """
    if len(ext) > 0:
        fmt = ext[1:].upper()  # remove the initial '.'
        if fmt == 'JPG':
            fmt = 'JPEG'
    else:  # no extension given: save as .png
        fmt = 'PNG'
    return fmt

def splitEndDigits(s):
    """Split a string in any prefix and a numerical end sequence.

    A string like 'abc-0123' will be split in 'abc-' and '0123'.
    Any of both can be empty.
    """
    i = len(s)
    if i == 0:
        return ( '', '' )
    i -= 1
    while s[i].isdigit() and i > 0:
        i -= 1
    if not s[i].isdigit():
        i += 1
    return ( s[:i], s[i:] )
