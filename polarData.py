
"""
Wrapper for atoring polar data taken of a microphone along with metadata about the experiment
@author: Tony Terrasa
"""
from __future__ import print_function
from pythonAudioMeasurements.pyStep import Stepper
from pythonAudioMeasurements.audioSample import audioSample
from math import atan
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# python 3 compatibility
import sys
if sys.version_info[0] < 3: 
    import cPickle as pickle
elif sys.version_info[0] >= 3: 
    import pickle

#import sys
#sys.path.append('../')

"""
To do:
> generate a polar plot for the given data
    > inputs - frequencies to be shown - this shoul dhave some set of default values
> considering changing the behavior of get_item to output the data raray from audioSample
Done:
> replace angle in one place with data from another reflected across some axis
> change or remove frequencies in any format for all angles
> get the values of the audio sample array at a given value
> save the current polarData instance to a pickle (usde to save the alterations that have been made)
> change frequency information for a specific angle
> change frequency informaiton for all angles
"""


class polarData:

    def __init__(self, angles=[], audioData=dict(), fs=44100):

        self._fs_rm  = []
        self.fs = fs
        self.angles = angles
        self.audioData = audioData


    def copy(self):
        """
        Returns a deep copy of this polarData instance.
        """

        audioDataCopy = dict()
        for theta, data in self.audioData.items():
            audioDataCopy[theta] = data.copy()

        return polarData(angles=self.angles.copy(), audioData=audioDataCopy, fs=self.fs)


    def addTo(self, pd):
        """
        Adds another polarData object to this one. Note that this operation 
        is done in place.
        
        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        pd  			| (polarData) to be added to this instance
        ---------------------------------------------------------------------
        
        """

        # make sure the angles match
        assert all([(pd.angles[i]-self.angles[i]) < 1e-8 for i in range(len(self.angles))]), \
            "polarDatas must have the same angles to be added"

        for theta in self.angles:
            self.audioData[theta] += pd.audioData[theta]


    def applyFilter(self, filt):
        """
        Applies the give filter to this polarData object in place by point-
        wise mulitplication
        
        
        ---------------------------------------------------------------------
        INPUTS
        ---------------------------------------------------------------------
        filt			| (audioSample) filter to be applied. Must have the
                        | same length as the frequency responses in the 
                        | this polarData object. Applies filter in current 
                        | type (in most cases, should be `f`) to this 
                        | instance in current type
        ---------------------------------------------------------------------
        
        
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        modifies this objcet in place
        ---------------------------------------------------------------------
        """

        assert len(self.audioData[self.angles[0]]) == len(filt), "input filter must be of same length as polarData"

        # type check
        if self.getType() != filt.type:
            print("WARNING: filter (%s) and polarData (%s) instance are of different types~"%(filt.type, self.getType()))

        # apply the filter by multiplication
        for theta in self.audioData.keys():
            self.audioData[theta] *= filt

        
    @staticmethod
    def fromPkl(filename, pickleAsAudioSample=False):


        """
        loads in data from polarPlot
        Args: 
        filename (str): path to pickled file from polarPlot
        Instance variables:
        self.filename = the given file location at the creation of the object
        self.angles = list of all the angles for which data was collected
        self.audioData = dictionary mapping angle to the audioSample collected    
        self._fs_rm = frequencies that have been removed from all audioSamples
                      polarData level    
        """

        pd = polarData()

        pd.filename = filename

        with open(filename, "rb") as file_:
            loadedFile = pickle.load(file_, encoding="latin1")


        # use for original-style pickles as originally recorded
        # not that this will only work if called from the 
        # pythonAudioMeasurements directory
        if pickleAsAudioSample: 

            pd.angles = loadedFile[0]["angles"] 

            # dictionary mapping int value of each angle to its appropriate audioSample
            pd.audioData = dict(zip(pd.angles, loadedFile[0]["measurements"]))

            pd.assertValidData()

        # recommended way of loading
        else:

            # angles
            pd.angles = loadedFile["angles"]

            # convert the tuples to audioSamples
            for i in range(len(loadedFile["measurements"])):
                asTuple = loadedFile["measurements"][i]
                loadedFile["measurements"][i] = audioSample(dataArray=asTuple[0], type=asTuple[1], Fs=asTuple[2],supress=True)

            # dictionary mapping int value of each angle to its appropriate audioSample
            pd.audioData = dict(zip(pd.angles, loadedFile["measurements"]))

        pd.removeDCOffset()

        return pd

    def assertValidData(self):
        """
        Asserts that all audioSamples are of the same sampling frequency and length (to preserve the integrity
        of the fourrier analysis across angles)
        """
        lengths = set()
        fs = set()

        for audioSamp in self.audioData.values():
            audioSamp.update()
            lengths.add(len(audioSamp.data))
            fs.add(audioSamp.fs)

        self.setType("Db")

        assert len(lengths) == 1, "error loading in pickle: varying DATA ARRAY LENGTHS. audioData must be in the format output of polar-measurement and all audioSamples must have the same sampling frequency and length of data array"
        assert len(fs) == 1, "error loading in pickle: varying SAMPLING FREQUENCIES. audioData must be in the format output of polar-measurement and all audioSamples must have the same sampling frequency and length of data array"
    
    def plotAngle(self, theta,both=False):
        
        if theta not in self.audioData:
            oldTheta = theta
            theta = self.getClosestAngle(theta)
            print("%d not in data set. using closest given angle: %d" % (oldTheta, theta))

        print("displaying plot for: %d"%theta)

        self.audioData[theta].plot(both=both)
    
    def plotFreqs(self, freqs=[], title="POLARDATA RESPONSE", fig=1, show=True):
        """
        Takes measured data as specified below and plots it using matplotlib
        on a polar axes
        Args:
        measured_data (dict): contains already-measured data in the format
                            {frequency1 : [(angle1 in degrees, amplitude1), ...], ....}
        """

        plt.figure(fig)

        # suplot on which all data will be places
        ax = plt.subplot(1, 1, 1, projection = "polar")

        


        ###
        # PULL OUT THE INDEX OF THE FREQS
        ###

        available_freqs = self.audioData[self.angles[0]].f()

        f_indeces = [-1]*len(freqs)
        f_difs = [1e12]*len(freqs)

        # find the closest frequency to each of the given input frequencies
        for i in range(len(available_freqs)):
            for j in range(len(freqs)):
                if abs(freqs[j] - available_freqs[i]) < f_difs[j]:
                    f_indeces[j] = i
                    f_difs[j] = abs(freqs[j] - available_freqs[i])

        # strings containing the names of the frequencies to be utilized
        legend = [str(int(available_freqs[i])) for i in f_indeces]

        # make sure all data in Db
        oldType = self.audioData[self.angles[0]].type 
        self.setType("Db")


        # loop through freqs
        for i in f_indeces:
            # loop through angles
            r = [self.audioData[phi].data[i].real for phi in self.angles]

            # print(freq, data)

            # unpack the data
            # comes out in tuple
            theta = tuple(self.angles)

            # add to ends to make plot loop
            # all the way around
            theta += (360,)
            r += (r[0],)


            theta = [t*np.pi/180 for t in theta] # convert to rad


            # linear interpolation between points
            """ perhaps a different interpolation would be better
                linear interpolation in polar leads to naturally out-swooping arcs"""
            f = interpolate.interp1d(theta, r)

            # create thetas
            theta_plot = np.arange(0,2*np.pi, 0.05)

            # using function tp interpolate r-points for
            # smoothness of output graphic
            r_plot = f(theta_plot)

            # plot this set of points
            ax.plot(theta_plot, r_plot)

        self.setType(oldType)

        # add graphic title
        ax.set_title(title)
        ax.grid(True) # turn on grid lines
        #ax.set_rticks(np.arange(0,50, 10)) # add tick marks
        ax.legend(legend, loc = "upper left") # create key
        #plt.savefig("fig_temp" + str())
        if show: plt.show()

    def replaceAnglesAxis(self, thetaLower, thetaUpper, thetaAxis=0, thetas =[]):
        """
        Replace the data contained in (thetaLower, thetaUpper) (inclusive) with symmetric data
        reflected over thetaAxis
        thetaLower (int): lower bound for the theta range to reflect
        thetaUpper  (int): upper bound for the theta rang to reflect
        thetaAxis (int - default=0): axis about which to reflect the given angle 
        """

        assert thetaLower >= 0 and thetaUpper >= 0, "Angles must be positive and cannot wrap around 0"
        assert not (thetaLower < thetaAxis < thetaUpper), "Axis of reflection within theta range given. Set reflection axis with argument thetaAxis "
    
        for theta in self.angles:

            # only edit angles within range
            if not thetaLower <= theta <= thetaUpper and theta not in thetas: continue

            
            thetaReplace = Stepper.validifyLoc(thetaAxis + (thetaAxis-theta)) # reflect angle theta over axis
            thetaReplace = self.getClosestAngle(thetaReplace) # find the closest angle to reflected angle

            # replace the given angle 
            self.audioData[theta] = self.audioData[thetaReplace]

    def replaceAngle(self, thetaReplace, thetaReplaceWith):
        """
        Sets the audioSample at thetaReplace to copy of the audioSample currently 
        at thetaReplaceWith 
        """
        assert (thetaReplace in self.angles), "given remove angle not in this polarData instance use polarData.getAngles() to see list of angles"
        assert (thetaReplaceWith in self.angles), "given replace angle not in this polarData instance use polarData.getAngles() to see list of angles"

        self.audioData[thetaReplace] = self.audioData[thetaReplaceWith].copy()
        
    def removeFreqsAtTheta(self, theta, freqs=[], freqRange=[]):
        """
        Uses changeFreqs to remove the given frequencies from the data set
        """
        self.changeFreqs("rm", theta, freqs=freqs, freqRange=freqRange)

    def changeFreqsAtTheta(self, value, theta, freqs=[], freqRange=[], mode=None):
        """
        Uses the audioSample.changeFreqs(self, value, freqs=[], freqRange=[-1,-1], dbOnly=False)
        to adjust values of specifed frequencies within the data at all angles. Autimatically 
        changes data to dB for change and then converts back to original type. 
        mode (str): 
            > if "f" - input value is in complex amplitude - converts value to polar coordinates before
              changing
            > if "db-only" - input is meant to effect the magnitude of a frequency only (not the phase)
            > if None, it assumes input value is given in db format
        Returns:
         > frequencies changed on this pass through the audioSamples
        """

        # convert input value if given in complex amplitude
        if mode == "f": value = complex(abs(value), atan(value.imag/value.real))

        theta = self.getClosestAngle(theta)

        audioSamp =  self.audioData[theta]
        
        _type = audioSamp._type # to convert back to 

        audioSamp.toDb() # ensure data in mag, phase domain 

        changed = audioSamp.changeFreqs(value, freqs=freqs, freqRange=freqRange, dbOnly=(mode=="db-only"))

        audioSamp.type = _type # convert back to original type

        return changed


    def changeFreqs(self, value, thetas=[], thetaRange=[-1,-1], freqs=[], freqRange=[], mode=None):

        """
        changes the frequencies of the audioSamples at the given angles, which can be input in exact angles or
        can be input as a range
        if the no thetas nor a range are given, the changes will be applied to the audioSamples at all thetas
        """

        changed = set()

        # if no angles given, it will alter the frequencies for all thetas
        if not thetas and thetaRange == [-1,-1]: thetaRange = [0,360]

        lowerTheta, upperTheta = thetaRange[:2]

        for theta in self.angles:
            
            if theta in thetas or lowerTheta <= theta <= upperTheta:                
                changed.update(self.changeFreqsAtTheta(value, theta, freqs=freqs, freqRange=freqRange, mode=mode))

        if value == "rm" or abs(value) < 1e-16:
            self._fs_rm.extend(changed)

        return changed

    def removeDCOffset(self):
        for theta in self.angles:
            self.audioData[theta].removeDCOffset()

    def setAngle(self, theta, audioSample):
        assert theta in self.angles, "given angle not in this audioSample use audioSample.getAngles() to see a lit of containined angles"
        assert self.audioData[0].fs == audioSample.fs and len(self.audioData[0]) == len(audioSample), "all audiosamples in a polarData instance must have the same fs and length"
        self.audioData[theta] = audioSample
            
    def setType(self, _type):
        for audioSamp in self.audioData.values(): audioSamp.setType(_type)

    def getAngles(self):
        return self.angles

    def getFreq(self, theta=0):
        return self.audioData[theta].f()

    def f(self): return self.getFreq()

    def getRemoved(self):
        return self._fs_rm

    def getData(self, theta=None):
        """
        Get audioSample data for this object at specified angle. 
        If no angle specified, all of the data will be returned. 
        """
        if theta is None: return self.audioData

        theta = self.getClosestAngle(theta)
        return self.audioData[theta]


    def __getitem__(self, theta):
        return self.getData(theta)

    def __setitem__(self, theta, val):
        self.audioData[theta] = val

    def getType(self):
        return self.audioData[0].type
    
    def getClosestAngle(self, theta):
        """
        Returns the value of the closest angle to theta in the dataset in degrees
        """

        if theta in self.angles: return theta

        difference = [abs(theta-a) for a in self.angles]
        return self.angles[difference.index(min(difference))]

    def to2dArray(self):
        """
        Returns a 2-d numpy array of the data in this array
        ---------------------------------------------------------------------
        OUTPUTS
        ---------------------------------------------------------------------
        data			| (numpy.array-2d) such that data[i] contains the 
                        | data array corresponding to angles[i]
        ---------------------------------------------------------------------
        """

        data = np.array([self.audioData[theta].data for theta in self.angles])

        return data

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)



            