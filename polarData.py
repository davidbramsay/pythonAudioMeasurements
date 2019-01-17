import cPickle as pickle
from pythonAudioMeasurements import pyStep
from pythonAudioMeasurements.audioSample import audioSample


class PolarData:

    def __init__(self, filename):

        with open(filename, "rb") as file_:
            loadedFile = pickle.load(file_)
        
        self.angles = loadedFile[0]["angles"] 

        # dictionary mapping int value of each angle to its appropriate audioSample
        self.audioData = {a : m for a,m in zip (self.angles, loadedFile[0]["measurements"])}
    
    def plotAngle(self, theta):
        
        if theta not in self.audioData:
            theta = self.getClosestAngle(theta)
            print str(theta) + " not in data set. using closest given angle: %d" % theta
            


        #### todo : add function to plot both the time and frequency responces in one figure

        self.audioData[theta].toDb()
        self.audioData[theta].plot()

        self.audioData[theta].toTime()
        self.audioData[theta].plot()

    def replaceBadAngles(self, thetaLower, thetaUpper, thetaAxis=0):
        """
        Replace the data contained in theta1, theta2 with symmetric data
        reflected over thetaAxis

        must input angles as positive numbers
        cannot wrap around zero (cannot do a range of (-30, 30))
        """

        assert thetaLower >= 0 and thetaUpper >= 0, "Angles must be positive and cannot wrap around 0"
        assert not (thetaLower < thetaAxis < thetaUpper), "Axis of reflection within theta range given. Set reflection axis with argument thetaAxis "
    
        for theta in self.angles:

            # only edit angles within range
            if not thetaUpper <= theta <= thetaUpper: continue

            
            thetaReplace = pyStep.validifyLoc(thetaAxis + (thetaAxis-theta)) # reflect angle theta over axis
            thetaReplace = getClosestAngle(thetaReplace) # find the closest angle to reflected angle

            # replace the given angle 
            self.audioData[theta] = self.audioData[thetaReplace]



    def setFreqs():
        """
        eliminate give frequencies form the data set
        """
    

    def getAngles(self):
        return self.angles

    def getFreq(self, theta=0):
        return self.audioData[theta].f()
        

    def checkAngle(self, theta):
        if theta not in self.audioData:
            print str(theta) + " not in data set. using closest given angle"
            return self.getClosestAngle(theta)

    def getClosestAngle(self, theta):

        if theta in self.angles: return theta

        difference = [abs(theta-a) for a in self.angles]
        return self.angles[difference.index(min(difference))]



            

        

