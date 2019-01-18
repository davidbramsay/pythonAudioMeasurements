import cPickle as pickle
import pyStep
from audioSample import audioSample
from math import atan


class polarData:

    def __init__(self, filename):


        """
        loads in data from polarPlot

        Args: 
        filename (str): path to pickled file from polarPlot

        Instance variables:
        self.filename = the given file location at the creation of the object
        self.angles = list of all the angles for which data was collected
        self.audioData = dictionary mapping angle to the audioSample collected        
        """

        self.filename = filename

        with open(filename, "rb") as file_:
            loadedFile = pickle.load(file_)
        
        self.angles = loadedFile[0]["angles"] 

        # dictionary mapping int value of each angle to its appropriate audioSample
        self.audioData = dict(zip (self.angles, loadedFile[0]["measurements"]))
    
    def plotAngle(self, theta,both=False):
        
        if theta not in self.audioData:
            oldTheta = theta
            theta = self.getClosestAngle(theta)
            print "%d not in data set. using closest given angle: %d" % (oldTheta, theta)            


        #### todo : add function to plot both the time and frequency responces in one figure

        self.audioData[theta].plot(both=both)

    def replaceBadAngles(self, thetaLower, thetaUpper, thetaAxis=0, thetas =[]):
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

            
            thetaReplace = pyStep.validifyLoc(thetaAxis + (thetaAxis-theta)) # reflect angle theta over axis
            thetaReplace = getClosestAngle(thetaReplace) # find the closest angle to reflected angle

            # replace the given angle 
            self.audioData[theta] = self.audioData[thetaReplace]

    
    def removeFreqs(self, freqs=[], freqRange=[]):
        """
        Uses changeFreqs to remove the given frequencies from the data set
        """
        self.changeFreqs("rm", freqs=freqs, freqRange=freqRange)
    
    def changeFreqs(self, value, freqs=[], freqRange=[], mode=None):
        """
        Uses the audioSample.changeFreqs(self, value, freqs=[], freqRange=[-1,-1], dbOnly=False)
        to adjust values of specifed frequencies within the data at all angles. Autimatically 
        changes data to dB for change and then converts back to original type. 

        mode (str): 
            > if "f" - input value is in complex amplitude - converts value to polar coordinates before
              changing
            > if "db-only" - input is meant to effect the magnitude of a frequency only (not the phase)
        """

        # don't convert all data into f for frequency changes
        # convert input value instead
        if mode == "f": value = complex(abs(value), atan(value.imag/value.real))

        for audioSamp in self.audioData.values():
            _type = audioSamp._type # to convert back to 

            audioSamp.toDb() # ensure data in freq domain 

            audioSamp.changeFreqs(value, freqs=freqs, freqRange=freqRange, dbOnly=(mode=="db-only"))

            audioSamp.type = _type # convert back to original type
            

    def setType(self, _type):
        for audioSamp in self.audioData.values(): audioSamp.type = _type

    def getAngles(self):
        return self.angles

    def getFreq(self, theta=0):
        return self.audioData[theta].f()
        

    def checkAngle(self, theta):
        if theta not in self.audioData:
            print str(theta) + " not in data set. using closest given angle"
        
        return self.getClosestAngle(theta)


    def getClosestAngle(self, theta):
        """
        Returns the value of the closest angle to theta in the dataset in degrees
        """

        if theta in self.angles: return theta

        difference = [abs(theta-a) for a in self.angles]
        return self.angles[difference.index(min(difference))]



            

        

