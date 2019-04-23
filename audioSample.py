# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:09:25 2015

@author: davidramsay
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import audioExtras as ae
import copy

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
        self._data = dataArray
        self._fs = Fs
        self._fs_rm = set()
        assert (type.lower() in ("t", "f", "db")), "type invalid, use t, f, or db"
        self._type = type.lower()

        if (type =="t"):
            self._tLength = len(self._data)
        else:
            self._tLength = 2* len(self._data) - 1
            print ("make sure data is single sided, as " +
            "described in np.fft.rfft doc.  Reverse " +
            "transform to time is non-deterministic, " +
            "it will be assumed that the time domain " +
            "length of the signal is 2*len(data)-1 ")


    def copy(self):
        """
        Returns a new audioSample instance that is a copy of self
        The data, type, and removed frequencies will be preserved

        data, and rempoved frequencies arrays are deepcopied
        """
        newAudioSample = audioSample(copy.deepcopy(self._data), type=self.type, Fs=self._fs)
        newAudioSample._fs_rm = copy.deepcopy(self._fs_rm)
        return newAudioSample


    def f(self):
        #return frequencies of samples
        if self._type=="t":
            print "watch out, your raw data is in time units!"

        try:
            return self.freqs
        except:
            self.freqs = np.linspace(0, self._fs/2, self._tLength//2 + 1)
            return self.freqs


    def t(self):
        #return times of samples
        if self._type in ("f", "db"):
           print "watch out, your data is in freq units!"

        return np.linspace(0.0, (float(self._tLength)-1.0)/self._fs, self._tLength)


    def __len__(self): return len(self._data)

    def __iter__(self):
        for val in self._data:
            yield val


    def update(self):
        variables = vars(self).keys()
        if "_type" not in variables: self._type = vars(self)["type"]
        if "_data" not in variables: self._data = vars(self)["data"]
        if "_fs" not in variables: self._fs = vars(self)["fs"]
        if "_fs_rm" not in variables: self._fs_rm = set()


    @property
    def data(self): return self._data


    @property
    def fs(self): return self._fs


    @property
    def type(self): return self._type

    @property
    def removed(self): return self._fs_rm


    @fs.setter
    def fs(self, value):
        raise Exception('NOT IMPLEMENTED: Changing Fs requires intelligent upsampling/decimation.')


    @data.setter
    def data(self, value):
        raise Exception('Prefer creating a new audiosample to setting data explicitly, this is a bad idea.')


    @type.setter
    def type(self, value, verbose=False):
        assert (value.lower() in ("t", "f", "db")), "type invalid, use t, f, or db"

        new_type = value.lower()
        current_type = self._type

        if new_type != current_type:
            if new_type == "t":
                if verbose: print 'converted to time'
                self.toTime()
            elif new_type == "f":
                if verbose: print 'converted to freq'
                self.toFreq()
            elif new_type == "db":
                if verbose: print 'converted to db'
                self.toDb()
            else:
                raise TypeError("instance.type is invalid!")
        else:
            if verbose: print 'already of that type'



    def toTime(self, verbose=False):
        if (self._type == "f"):
            self._data = np.fft.irfft(self._data, self._tLength)
            self._type = "t"

        elif (self.type == "db"):
            self.toFreq()
            self.toTime()

        elif (self.type == "t"):
            if verbose: print "already in time"

        else:
            raise TypeError("instance.type is invalid!")


    def toFreq(self, verbose=False):
        if (self._type == "t"):
            self._data = np.fft.rfft(self._data)
            self._type = "f"

        elif (self._type == "db"):
            unDBed = pow(10, self._data.real/20.0)
            self._data = unDBed*np.cos(self._data.imag) + 1j*unDBed*np.sin(self._data.imag)
            self._type = "f"

        elif (self._type == "f"):
            if verbose: print "already in freq"

        else:
            raise TypeError("instance.type is invalid!")


    def toDb(self, verbose=False):
        if (self._type == "f"):
            mag = 20*np.log10(np.abs(self._data))
            phase = np.angle(self._data)
            self._data = mag+(1j*phase)
            self._type = "db"

        elif (self._type == "t"):
            self.toFreq()
            self.toDb()

        elif (self._type == "db"):
            if verbose: print "already in db"

        else:
            raise TypeError("instance.type is invalid!")


    def plot(self, both=False):
        #plot the signal in the current domain

        fig = plt.subplot(1,1,1)

        if both:
            fig = plt.subplot(2,1,1)
            _type = self._type

            self.toTime()
            plt.plot(self.t(), self._data)
            plt.title("Time Domain Plot")
            plt.grid(True)
            plt.xlabel('time (s)')
            plt.ylabel('magnitude')

            plt.subplot(2,1,2)
            self.toDb()
            plt.semilogx(self.f(), self._data.real)
            plt.title("Single Sided FFT Magnitude")
            plt.grid(True)
            plt.xlabel('freq (Hz)')
            plt.ylabel('dBFS')

            # gives space for axis labels
            plt.subplots_adjust(hspace=.75) 

            plt.show()

            # convert type back to whatever it was
            self.type = _type

            
        elif (self._type == "t"):
            plt.plot(self.t(), self._data)
            plt.title("Time Domain Plot")
            plt.grid(True)
            plt.xlabel('time (s)')
            plt.ylabel('magnitude')
            plt.show()

        elif (self._type == "f"):
            self.toDb()
            plt.semilogx(self.f(), self._data.real)
            plt.title("Single Sided FFT Magnitude")
            plt.grid(True)
            plt.xlabel('freq (Hz)')
            plt.ylabel('dBFS')
            plt.show()
            self.toFreq()

        else:
            plt.semilogx(self.f(), self._data.real)
            plt.title("Single Sided FFT Magnitude")
            plt.grid(True)
            plt.xlabel('freq (Hz)')
            plt.ylabel('dBFS')
            plt.show()



    def PDF(self, ac_couple=False):
        #plot the PDF
        def plotPDF():

            x = self.f()
            y = np.square(np.abs(self._data))

            if ac_couple: y[0]=0

            plt.plot(x,y)
            plt.xscale('symlog')
            plt.title("PSD")
            plt.grid(True)
            plt.autoscale()
            plt.xlabel('freq (Hz)')
            plt.ylabel('Power')
            plt.show()

        self.applyInDomain('f', plotPDF)


    def removeDCOffset(self, new_offset=0):
        #add or remove dc offset
        def setOffset():
            self._data = self._data - np.mean(self._data)
            self._data = self._data + new_offset

        self.applyInDomain('t', setOffset)

    
    def removeFreqs(self, freqs=[], freqRange=[-1,-1]):
        self.changeFreqs("rm", freqs=freqs, freqRange=freqRange)


    def changeFreqs(self, value, freqs=[], freqRange=[-1,-1], dbOnly=False):
        """
        Changes data values for given frequencies. 
        
        Frequencies can be given as discrete frequencies or as a range. If given both descrete
        values and a range are given both sets of frequencies will be adjusted. If a frequency is 
        neither in the signal nor within the given range, it will be ignored

        Data in type "f" cannot be set to 0. 

        If given value is floating point 0, then it will be counted as removing that frequency




        Args:
        value (int, float, complex, str): new data value for given frequency
            > using the "rm" value will set that frequency amplitude equal to 0
            > can be entered as real number or complex
        freqs (int, float, list): discrete frequencies to be altered
        freqRange (list): first two indexes used as lower bound and upper bound respectively. 
                          Range is inclusive on both ends
                          Defaults to (-1, -1) to which no frequencies will match
        dbOnly (bool): if true, just the magnitude of the frequency will be set to the real part of value, and the phase will be left alone

        Returns:
        changed
            > array of frequencies changed on THIS PASS
            
            
        """

        # ensure data is in freq domain
        assert self._type != "t", "data is in time domain. use audioSample.toFreq() or audioSample.toDB() to convert to frequency domain"


        # allows for single frequency input
        if isinstance(freqs, (int, float)): freqs = [freqs]

        # extract bounds for frequency range
        minF, maxF = freqRange[:2]


        if dbOnly:
            assert self._type == "db", "db frequency change attemped with data not in db. Use audioSample.toDB() to change data to db"
            
                    
        # allows for implied 0 imaginary part
        if not isinstance(value, complex) and value != "rm": value = complex(value, 0)

        assert self._type != "f" or abs(value) > 0, "cannot set complex amplitude to 0"

        # frequencies changed in this process
        changed = []

        
        set_value = 1e-12 if value == "rm" else complex(value)

        if value == "rm": self.toFreq()

        for i, f in enumerate(self.f()):
            if f in freqs or minF <= f <= maxF:

                changed.append(f)
                
                # store each frequency removed
                if abs(set_value) < 1e-15 or value == "rm": 
                    self._fs_rm.add(f)
                    

                # adjust mag, preserve phase
                if dbOnly: 
                    set_value = complex(set_value.real, self._data[i].imag) 
                    
                self._data[i] = set_value    

        return changed    

            



    def normalize(self):
        #normalize to [-1,1] for time data and 0dBFS for dB data
        assert(len(self._data)), 'must have data to normalize'
        if (self._type == "t"):
            assert (np.mean(self._data) < 1e-15), 'must not have DC component to be normalized, remove with removeDCOffset'
            print "normalizing time data to [-1, 1]"
            self._data = self._data / float(np.amax(np.abs(self._data)))
        elif (self._type == "db"):
            print "normalizing db data to 0dBFs"
            self._data.real = self._data.real - float(np.amax(self._data.real))
        else:
            print "not a normalizable type, please submit time domain or db data"


    def setVolume(self, volume=-6):
        #set volume relative to normalized [-1, 1] full-scale
        assert(len(self._data)), 'must have data to alter scaling'
        assert (volume <= 0), 'volume must be in relative dBFS to full-scale (<=0)'

        def setVol():
            assert (np.mean(self._data) < 1e-15), 'must not have DC component to set volume, remove with removeDCOffset'
            linear_gain = 10**(volume/20.0)
            self._data = self._data * linear_gain / float(np.amax(np.abs(self._data)))

        self.applyInDomain('t', setVol)


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

        temp = np.zeros(len(self._data), float)
        freqs = self.f()

        for i in range(len(temp)):
            bounds = ae.octaveSpacing(freqs[i], octSmooth)
            temp[i] = np.mean(self._data[np.where((freqs>=bounds[0]) & (freqs<=bounds[1]))].real)

        if destructive in ['y', 'Y', 'yes', 'YES']:
            self._data.real = temp
            print ("DESTRUCTIVE ACTION - this will change the magnitude data" +
            " stored irreversibly, and other functions will work but have no" +
            " intuitive meaning")
        else:
            return audioSample(temp, 'dB', self._fs)


    def hanning(self):
        #apply hanning window to the time domain signal
        def hannIt():
            self.data = self.data * np.hanning(self._tLength)

        self.applyInDomain('t', hannIt)


    def zeroPadStart(self, length=64):
        #apply zero pad to the beginning of the time domain signal
        def padIt():
            self._data = np.concatenate([np.zeros(length), self._data])
            self._tLength = len(self._data)

        self.applyInDomain('t', padIt)


    def zeroPadEnd(self, length=64):
        #apply zero pad to the end of the time domain signal
        def padIt():
            self._data = np.concatenate([self._data, np.zeros(length)])
            self._tLength = len(self.data)

        self.applyInDomain('t', padIt)


    def applyInDomain(self, domain, func):

        assert (domain in ['t','f','db']), 'domain must be t, f, or db'
        current_type = self._type
        undo_domain = False

        if domain != current_type:
            undo_domain = True
            if domain == "t": self.toTime()
            elif domain == "f": self.toFreq()
            elif domain == "db": self.toDb()

        func()

        if undo_domain:
            if current_type == "t": self.toTime()
            elif current_type == "f": self.toFreq()
            elif current_type == "db": self.toDb()

