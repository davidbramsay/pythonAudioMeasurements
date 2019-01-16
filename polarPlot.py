from audioMeasure import audioMeasure
from audioSample import audioSample
from pyfirmata import Arduino, util
from pyStep import Stepper
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import interpolate
import cPickle as pickle


class polarPlot(object):

    # as of right now, you pass in a pyStepper motor instance
    def __init__(self, usingMotor=None, makeMotor=False, board="/dev/cu.usbmodem1441", pins=[2,3,4,5], audioMeasure=None, testSignalSamples=8192, testSignalRepetitions=30, Fs=44100, channels=1):

        if audioMeasure:
            self.audioMeas = audioMeasure
        else:
            self.testSignalSamples = testSignalSamples
            self.testSignalRepetitions = testSignalRepetitions
            self.audioMeas= audioMeasure(Fs=Fs)
            self.audioMeas.pinkNoiseLoop(samples=testSignalSamples, repetitions=testSignalRepetitions)

        self.allFrequencies = None
        self.response = None
        self.board = None
        self.motor = None
        self.channels = channels

        # initiate and set up motor if needed
        if usingMotor:
            self.motor = usingMotor

        if makeMotor:
            self.board = Arduino(board)
            self.motor = Stepper(self.board, pins)


        # ensures self.allFrequencies gets set
        print("initializing. NOT MEASURING")
        self.measure(True)

    def makePlot(self, numMeasurements=4, measurementFrequencies=None): # toimplement: degreeMeasurements=None
        """
        Makes a polar plot of the default connected microphone

        Args: see polarPlot.collectData
        """
        data = self.collectData(numMeasurements=numMeasurements, measurementFrequencies=measurementFrequencies)
        self.plot(data)

    def getClosestFreq(self, freq):

        minDifference = 1000000000
        closestFreq = None

        # store the closest frequency to the first one
        for f in self.allFrequencies:
            if closestFreq is None: closestFreq = f

            if abs(f - freq) < minDifference:
                closestFreq = f
                minDifference = abs(f - freq)

        message = "Input frequency not found. Closest frequency to input will be used. \n"
        message += "Input: %d \n" % freq
        message += "Replaced by: %d \n" % closestFreq

        print(message)

        return closestFreq

    def measure(self, test=False):
        """
        Takes a pink noise measurement

        Returns:
        (list) containing frequency-domain data for the measurement
        """

        # run tests
        self.audioMeas.testAllChannels()
        self.audioMeas.calcTF()

        if test:
            self.audioMeas.plotFreqResp() #plot freq response of each channel
            self.audioMeas.plotImpulseResp() #plot IR of each channel

        # extract tf values note type is
        # audioSample
        audioSamp = self.audioMeas.tf[0]

        audioSamp.toDb()


        if self.allFrequencies is None:
            # contains a list of integers for all
            # the frequencies collected
            self.allFrequencies = [int(f) for f in audioSamp.f()]



        # return just a list containing the values of the
        # transfer function in freq units
        return audioSamp

    def collectData(self, numMeasurements=4, measurementFrequencies=None, degreeSeparation=None):
        """
        Args:
        numMeasurements (int): number of measurements to be taken evenly spaced around the circle
                               if motor in use, then with value 2, measurements will be taken at
                               0 and 180 degrees
        measurementFrequencies (list): frequencies to create a plot for.
                                        ## current;y: throws out any frequencies not measured
                                        ## future: calculate the closest frequency and
        """


        """ think this is unnecessary (handled in constructor) """
        if self.allFrequencies is None:
            # all frequencies will be collected in measure
            # this is because if this has not been set yet,
            # then a transfer function has not yet been calculated
            time.sleep(0.5)
            self.measure()



        if measurementFrequencies is not None:

            # ensures all inputted frequencies are integers
            # held as set
            measurementFrequencies = {int(f) for f in measurementFrequencies}

            toRemove = set()
            toAdd = set()

            # make sure all inputted frequencies will be measured
            # save ones not measured to be removed later
            for f in measurementFrequencies:
                if f not in self.allFrequencies:
                    toRemove.add(f)
                    toAdd.add(self.getClosestFreq(f))

            # remove unmeasured frequencies
            measurementFrequencies.difference_update(toRemove)
            measurementFrequencies.update(toAdd)

        # grabs 4 equally spaced (in the array) frequencies if none inputted
        # note allFrequencies always contains integers
        # held as set
        else:
            print("auto assigning frequencies...")
            measurementFrequencies = {self.allFrequencies[i] for i in range(0, len(self.allFrequencies), len(self.allFrequencies)//4)}
            print(measurementFrequencies)



        # contains the indexes in the input data that the given frequency's
        # data will appear
        indexes = {f : self.allFrequencies.index(f) for f in measurementFrequencies}

        # maps integers representing the frequencies data was collected at
        # to the storage of the turn data
        freqData = {f : [] for f in measurementFrequencies}


        # calculate the intervals if none given
        if not degreeSeparation:
            degreeSeparation = 360//numMeasurements


        #collect some relevant information from the user
        pickleName = raw_input("please enter a filename (without extension):")
        pickleFile = './data/' + str(pickleName) + '.pkl'

        micManufacturer = raw_input("please enter manufacturer/maker:")
        micModel = raw_input("please enter microphone model:")

        print 'axis of measurements:'
        print '\tfor MEMs mics, the short dimension closest to the port is facing 0 degrees.  For Electrets, 0 degrees will be along the axis that splits the pins' + \
              ', towards the "top" if neg/gnd pin on the right looking top down (through the mic).  Y runs along the 0-180 degree axis, X runs along the 90-270 ' + \
              'degree axis, (as if looking at a polar with 0 degrees on top, with axes labeled appropriately. Z runs through the axis of symmetry of the ' + \
              'microphone (positive towards the ceiling).\n\n\'x-y\' in plane of floor with mic/array facing upwards (primary direction you expect to be used).  Y=0 degrees.' + \
              '\n\'x-z\' is with positive/0-Y direction towards the ceiling, in the X-Z plane, with positive-Z facing at 0 degrees.'

        measureAxis = raw_input("please enter axis of measurement ('x-y', 'x-z' (default: x-y):")
        if not measureAxis: measureAxis = "x-y"

        offset = raw_input("please enter offset from 0,0 in rotation dimension, POS-Y = 0 degrees toward speaker (default: (0,0)):")
        if not offset: offset = (0, 0)

        location = raw_input("please enter the location of this measurement (default: MIT Anechoic Chamber, Blg 41):")
        if not location: location = "MIT Anechoic Chamber, Blg 41"

        speaker = raw_input("please enter the speaker used for this measurement (default: Equator Audio D5):")
        if not speaker: speaker = "Equator Audio D5"

        extraInfo = raw_input("please enter any additional description of this measurement:")

        raw_input('MAKE SURE THE DEVICE IS: (1) CENTERED IN FLOOR-CEILING DIM WITH CENTER OF SPEAKER, (2) ALIGNED ALONG 0 AXIS, (3) TURNTABLE IS FACING SHORT DIM, (4) DISTANCE' + \
              ' IS APPROX 2.5M, (5) SPEAKER IS LINEAR BUT LOUD.  OK?')

        timestamp = time.time()

        metadata = {
            'filename': pickleFile,
            'manufacturer': micManufacturer,
            'model': micModel,
            'measureAxis': measureAxis,
            'offset': offset,
            'location': location,
            'speaker': speaker,
            'description': extraInfo,
            'timestamp': timestamp,
            'degreeRes': degreeSeparation
        }

        polarData = {'angles': [], 'measurements': []}

        self.countDown(35)

        # loops through a whole turn
        # evenly dividing into numMeasurements steps
        # rounding down
        """potentially store the angles measured in a separate list to avoid
            repetitious data/excessive memory usage"""
        for degrees in range(0,360, degreeSeparation):


            # set the motor to the correct position
            if self.motor:
                self.motor.set_position(degrees)
            else:
                # count down and allow for the operator
                # to manually move the motor
                self.countDown(3)

            audiomeas = self.measure()

            polarData['angles'].append(degrees)
            polarData['measurements'].append(audiomeas)
            pickle.dump((polarData, metadata), open(metadata['filename'], 'wb'))

            for freq in measurementFrequencies:

                # index in data this frequency's amp should be located at
                index = indexes[freq]


                # add (angle in degrees, real part of amplitude) to this frequency's data list
                freqData[freq].append((degrees, audiomeas.data[index].real))

        # allow for sucessive plots to be made without
        # having to reposition
        self.motor.set_position(0)

        # coordinates now contains the appropriate data,, now need to plot them
        return freqData

    def plot(self, measured_data):
        """
        Takes measured data as specified below and plots it using matplotlib
        on a polar axes

        Args:
        measured_data (dict): contains already-measured data in the format
                            {frequency1 : [(angle1 in degrees, amplitude1), ...], ....}
        """

        # suplot on which all data will be places
        ax = plt.subplot(1, 1, 1, projection = "polar")

        # collects strings of the frequencies plotted
        # to create a legend at the end
        legend = []

        #just 3rd one for now plot the first one for
        for freq in measured_data:


            # add this frequency to the key to be shown
            # on the output grid
            legend.append(str(freq))

            data = measured_data[freq]

            # print(freq, data)

            # unpack the data
            # comes out in tuple
            theta, r = zip(*data)

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

        # add graphic title
        ax.set_title("MICROPHONE RESPONSE AT VARIOUS FREQUENCIES")
        ax.grid(True) # turn on grid lines
        ax.set_rticks(np.arange(0,50, 10)) # add tick marks
        ax.legend(legend, loc = "upper left") # create key
        plt.savefig("fig_temp" + str())
        plt.show()

    def countDown(self, sec):
        """
        Counts down from the given number of seconds, allowing hand-adjustment of  the microphone

        Args:
        sec (int): number of seconds to count-down
        """
        print("\n\n Leaving time to move the microphone. You have....")
        while sec > 0:
            print(sec)
            time.sleep(1)
            sec -= 1
        print("\n\n")


