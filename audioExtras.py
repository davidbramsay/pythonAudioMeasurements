# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:15:45 2015

@author: davidramsay
"""

import math
import numpy as np
import sys
import cStringIO
import struct
from itertools import izip_longest

"""
helper functions for audio interactions.

octaveSpacing(f0, octave)- returns f1 & f2, spaced octave apart and centered around f0
normalize(floatVals) - normalize a float array to [-1, 1]

floatsToWavBinary(array,chunk) - convert floats to an array of int16 chunks (size chunk) to be played
int16toFloat(array) - convert array of int16s to floats

disablePrint() - supresses all screen output
enablePrint() - re-enables screen output

grouper(iterable, n, fillvalue) - return an array of iterable in chunks of size n, and
fills leftover space in last chunk with fillvalue (default None)

"""

def octaveSpacing(f0, octave):
    #returns f1,f2 spaced octave and centered at f0 logarithmically
    f1 = f0 / (math.sqrt(10 ** (octave*np.log10(2))))
    f2 = f0 * (math.sqrt(10 ** (octave*np.log10(2))))
    #ratio of two freqs = 10 ** (octave diff * np.log10(2))
    #center freq = math.sqrt(freq1*freq2)
    return f1, f2
    
    
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def normalize(floatVals):
    floatVals = floatVals/max(abs(floatVals))
    return floatVals


def floatsToWavBinary(array,chunk = 1024):
    max_amplitude = 32767.0
    array = array*max_amplitude
    array = array.astype(int)

    binArray = []
    for x in grouper(array, chunk, 0):
        binArray.append(struct.pack("%sh" % len(x), *x))

    return binArray


def int16ToFloat(array):
    return array/32767.0    
  
  
def disablePrint():
    sys.stdout = cStringIO.StringIO()
 
   
def enablePrint():
    sys.stdout.flush() 
    sys.stdout = sys.__stdout__