# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:09:25 2015

@author: davidramsay
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import audioExtras as ae

"""
audioSample stores audio with information about its representation (time, freq, db) to easily
manipulate it.

time is in time
freq is in complex frequency
db is in mag/phase representation, in db


initialize by calling:
a = audioSample(np.array([1,2,4,5]), "f", 44100)  #f for single sided freq domain, 44100 for samplerate
a = audioSample(np.array([1,2,4,5]), "t", 48000)  #t for single sided freq domain, 44100 for samplerate
a = audioSample(np.array([1,2,4,5]))  #assumes 44100, time-domain

audioSample.data returns the raw data in whatever form is specified by type
audioSample.type returns type t, f, or db.  t is real, f is complex, db is complex [mag + j(phase)]
audioSample.fs returns the sampleRate


available class methods are:

audioSample.f() - get frequency values array [0, 20, 40 ...] Hz
audioSample.t() - get time value array [0, 1, 2, 3] Sec
audioSample.toTime() - put data in time domain
audioSample.toFreq() - put data in freq domain
audioSample.toDb() - put data in dB

audioSample.plot() - plot the data in whatever form it's in
audioSample.PDF() - plot the PDF of the data

audioSample.normalize() - normalizes time signal to [-1,1] and dB signal to 0dBFS

audioSample.hanning() - apply hanning window to the *time-domain* version of the data
audioSample.zeroPadStart(length) - zero pad (<length> zeros) the start of the *time-domain* version of the data
audioSample.zeroPadEnd(length) - zero pad (<length> zeros) the end of the *time-domain* version of the data

[The following two methods put the data in dB (no others alter the type), and are destructive if flag is 'y'.
This replaces dB data with smoothed data.  If 'n', an audioSample object with smoothed data is returned.]

audioSample.smoothFFT(octSmooth, destructive) - smooth once with octSmooth resolution (.10 or .33, for instance)
audioSample.doubleSmooth(octSmooth, destructive) - smooth twice with octSmooth resolution


Potential to add/change:

-fix destructive to be a True/False flag
-rotate data in time
-setters/getters for data/attributes
-other windows for windowing
-stereo/multichannel audio support (with simple .channels() method to check channel count)
-double check/accept and format any input arrays into proper np array size (1-D only right now)
-frame iterators (give a frameSize and get back iterator object)
-sample rate conversion
-interpolation of different types, ability to ask for any time or freq and get interpolated value linear, spline
-up- and down- sample
-overload addition, subtraction, multiplication, division of these objects to make sense

-change functions like hanning and zeropad to only work when it's time domain, instead of applying in time domain
and switching back to current representation?  more clunky for user but more sensical paradigm...


"""

class audioSample(object):


    def __init__(self, dataArray = [], type = "t", Fs = 44100):
        self.data = dataArray
        self.fs = Fs

        if type in ("t", "f", "db", "T", "F", "DB", "dB"):
            self.type = type.lower()
        else:
            raise NameError("type invalid, use t, f, or db")

        if (type =="t"):
            self._tLength = len(self.data)
        else:
            self._tLength = 2* len(self.data) - 1
            print ("make sure data is single sided, as " +
            "described in np.fft.rfft doc.  Reverse " +
            "transform to time is non-deterministic, " +
            "it will be assumed that the time domain " +
            "length of the signal is 2*len(data)-1 ")


    def f(self):
        #return frequencies of samples
        if self.type=="t":
            print "watch out, your raw data is in time units!"

        return np.linspace(0, self.fs/2, self._tLength//2 + 1)


    def t(self):
        #return times of samples
        if self.type in ("f", "db"):
           print "watch out, your data is in freq units!"

        return np.linspace(0.0, (float(self._tLength)-1.0)/self.fs, self._tLength)


    def toTime(self):
        if (self.type == "f"):
            self.data = np.fft.irfft(self.data, self._tLength)
            self.type = "t"

        elif (self.type == "db"):
            self.toFreq()
            self.toTime()

        elif (self.type == "t"):
            print "already in time"
        else:
            raise TypeError("instance.type is invalid!")


    def toFreq(self):
        if (self.type == "t"):
            self.data = np.fft.rfft(self.data)
            self.type = "f"

        elif (self.type == "db"):
            unDBed = pow(10, self.data.real/20.0)
            self.data = unDBed*np.cos(self.data.imag) + 1j*unDBed*np.sin(self.data.imag)
            self.type = "f"

        elif (self.type == "f"):
            print "already in freq"
        else:
            raise TypeError("instance.type is invalid!")


    def toDb(self):
        if (self.type == "f"):
            mag = 20*np.log10(np.abs(self.data))
            phase = np.angle(self.data)
            self.data = mag+(1j*phase)
            self.type = "db"

        elif (self.type == "t"):
            self.toFreq()
            self.toDb()

        elif (self.type == "db"):
            print "already in db"
        else:
            raise TypeError("instance.type is invalid!")


    def plot(self):
        #plot the signal in the current domain
        if (self.type == "t"):
            plt.plot(self.t(), self.data)
            plt.title("Time Domain Plot")
            plt.grid(True)
            plt.xlabel('time (s)')
            plt.ylabel('magnitude')
            plt.show()

        elif (self.type == "f"):
            self.toDb()
            plt.semilogx(self.f(), self.data.real)
            plt.title("Single Sided FFT Magnitude")
            plt.grid(True)
            plt.xlabel('freq (Hz)')
            plt.ylabel('dBFS')
            plt.show()
            self.toFreq()

        else:
            plt.semilogx(self.f(), self.data.real)
            plt.title("Single Sided FFT Magnitude")
            plt.grid(True)
            plt.xlabel('freq (Hz)')
            plt.ylabel('dBFS')
            plt.show()


    def PDF(self):
        #plot the PDF
        def plotPDF():
            plt.semilogx(self.f(), np.abs(np.square(self.data)))
            plt.title("PSD")
            plt.grid(True)
            plt.xlabel('freq (Hz)')
            plt.ylabel('Power')
            plt.show()

        if (self.type == "t"):
            self.toFreq()
            plotPDF()
            self.toTime()

        elif (self.type == "f"):
            plotPDF()

        elif (self.type == "db"):
            self.toFreq()
            plotPDF()
            self.toDb()


    def normalize(self):
        #normalize to [-1,1] for time data and 0dBFS for dB data
        if (self.type == "t"):
            print "normalizing time data to [-1, 1]"
            self.data = self.data / float(np.amax(np.abs(self.data)))
        elif (self.type == "db"):
            print "normalizing db data to 0dBFs"
            self.data.real = self.data.real - float(np.amax(self.data.real))
        else:
            print "not a normalizable type, please submit time domain or db data"


    def doubleSmooth(self, octSmooth, destructive = 'y'):
        #call smoothing function twice with above octave
        if destructive in ['y', 'Y', 'yes', 'YES']:
            self.smoothFFT(octSmooth)
            self.smoothFFT(octSmooth)

        else:
            print "CREATED COPY of original samples"
            temp = copy.deepcopy(self)
            temp.smoothFFT(octSmooth)
            temp.smoothFFT(octSmooth)             
            return temp
  
          
    def smoothFFT(self, octSmooth, destructive = 'y'):
        #convert data to dB and smooth magnitude data, leaving phase
        #this is destructive of magnitude data in the audioSample class unless specified not to be,
        #will return a new audioSample if 'nondestructive'
        self.toDb()

        temp = np.zeros(len(self.data), float)
        freqs = self.f()

        for i in range(len(temp)):
            bounds = ae.octaveSpacing(freqs[i], octSmooth)
            temp[i] = np.mean(self.data[np.where((freqs>=bounds[0]) & (freqs<=bounds[1]))].real)
        
        if destructive in ['y', 'Y', 'yes', 'YES']:
            self.data.real = temp
            print ("DESTRUCTIVE ACTION - this will change the magnitude data" +
            " stored irreversibly, and other functions will work but have no" +
            " intuitive meaning")
        else:
            return audioSample(temp, 'dB', self.fs)


    def hanning(self):
        #apply hanning window to the time domain signal
        def hannIt():
            self.data = self.data * np.hanning(self._tLength)

        if (self.type == "t"):
            hannIt()

        elif (self.type == "f"):
            self.toTime()
            hannIt()
            self.toFreq()

        elif (self.type == "db"):
            self.toTime()
            hannIt()
            self.toDb()
    
        
    def zeroPadStart(self, length=64):
        #apply zero pad to the beginning of the time domain signal
        def padIt():
            self.data = np.concatenate([np.zeros(length), self.data])            
            self._tLength = len(self.data)

        if (self.type == "t"):
            padIt()

        elif (self.type == "f"):
            self.toTime()
            padIt()
            self.toFreq()

        elif (self.type == "db"):
            self.toTime()
            padIt()
            self.toDb()
   
     
    def zeroPadEnd(self, length=64):
        #apply zero pad to the end of the time domain signal
        def padIt():
            self.data = np.concatenate([self.data, np.zeros(length)])            
            self._tLength = len(self.data)

        if (self.type == "t"):
            padIt()

        elif (self.type == "f"):
            self.toTime()
            padIt()
            self.toFreq()

        elif (self.type == "db"):
            self.toTime()
            padIt()
            self.toDb()