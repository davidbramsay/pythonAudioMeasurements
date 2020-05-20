"""

When microphone measurements where initially taken, the audioSamples
where stored in a pickle in such a way that they could not be 
reloaded outside the pythonAudioMeasurements directory. This file
corrects that, storing all of the data in pure Python formats

"""
import numpy as np
import pickle
from pythonAudioMeasurements.audioSample import audioSample
from pythonAudioMeasurements.polarData import polarData
import os

# whether to save the converted objects
CONVERT = True 

# whether to compare the two representations of the same audioSample
# which should be the same
COMPARE = False 

directory = "/home/terrasa/UROP/polar-measurement/data/19_Jan15" 

for filename in os.listdir(directory):

    # full path of the pickle
    full_path = directory + "/" + filename
    # path to the new location
    full_mod_path = directory + "_fixedpkls/" + filename
    # extension
    extension = filename.split(".")[-1]
    print(extension)
    print(full_path)
    print(full_mod_path)

    # ensure that this is a pickle object that can be loaded
    if extension != "pkl":
        print("not a pickle, moving on")
        continue


    # some sessions of collecting data had a baseline for the 
    # microphone and the room collected. These were stored 
    # pickles as just one audioSample and had to be converted 
    # differently
    if filename[:4] == "meas":
        with open(full_path, "rb") as file_:
            loadedFile = pickle.load(file_, encoding="latin1")
        m = audioSample(loadedFile).toStorageTuple()
        with open(full_mod_path, "wb") as f_:
            pickle.dump(loadedFile, f_)

        continue

    with open(full_path, "rb") as file_:
        # dictionary with "measurements" -> [list of audioSamples]
        loadedFile = pickle.load(file_, encoding="latin1")[0]

    # change the value stored for the key "measurements" to a list
    # of "storagee tuples" as specified in audioSample
    m = loadedFile["measurements"]
    new_measurements = []

    for i in range(len(m)):
        new_measurements.append(m[i].toStorageTuple())
        # print(m[i].type)
        # print(new_measurements[i][1])
        # print(m[i].data[:20])
        # print(new_measurements[i][0][:20])

    loadedFile["measurements"] = new_measurements

    if COMPARE:
        tester_index = 3
        m[tester_index].plot(both=True)
        tester = loadedFile["measurements"][tester_index]
        tester_asamp = audioSample(tester[0], tester[1], tester[2])

        tester_asamp.plot(both=True)


    if not CONVERT: continue

    with open(full_mod_path, "wb") as f_:
        pickle.dump(loadedFile, f_)

