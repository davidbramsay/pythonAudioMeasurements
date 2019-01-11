from pyfirmata import Arduino, util
import time

# board = Arduino("/dev/cu.usbmodem1441")

class Stepper(object):

    def __init__(self, board, pins, stepsPerRot=512, sleepTime=0.0002):
        """
        Args:
        pins (iterable of ints): containing the pins to be used for the
                                 motor in order
        board (Arduino): must be a properly initialized Arduino board
                         via pyfirmata
        stepsPerRot (int): number of steps per rotation
        sleepTime (float): desired time in seconds between phases in
                            motor turn
        """

        # check pins input
        assert (len(pins) == 4), "Must enter exactly 4 pins for this motor"
        assert (all([str(pin).isdigit() for pin in pins])), "Pins must be given as digits"

        # string labels for pin declaration
        # strings are to pyfirmata standard
        self.pinLabels = ["d:" + str(pin) + ":o" for pin in pins]

        # maps strings corresponding to directions to helper functions
        # as necessary to drive in the given direction (see Stepper.turn)
        self.possibleDrives = {"CW": Stepper.CW_drive, "CCW": Stepper.CCW_drive}
        self.location = 0 # position of motor in degrees


        self.board = board
        self.pins = [self.board.get_pin(pin) for pin in self.pinLabels] # declaration of pin objects

        self.stepsPerRot = stepsPerRot*8 # times 8 because there are 8 phases per step
        self.sleepTime = sleepTime  # seconds



    def set_position(self, location):
        """
        Args:
        location (int, float): angular position to set motor to

        Returns:
        (int): location after movement
        """

        # make sure 0<=location<360
        location = Stepper.validifyLoc(location)

        cw = location - self.location
        ccw = cw - 360

        # optimization of direction
        degToTurn = ccw if min(abs(ccw), abs(cw)) == abs(ccw) else cw

        self.turn(degToTurn)

        return self.location

    def turn(self, degrees, direction="CCW"):
        """
        Args:
        degerees (int, float): number of degrees to be turned
        direction (str): direction to turn the motor, defaults for clockwise
                         valid entries: 'CW' or 'CCW'
                         Direction decided by looking down on the motor

        Return:
        (int): location after movement

        Turns the motor the given number of degrees in the given direction.
        If given a negative degrees, it will turn in the opposite direction.

        Driving follows the bellow pattern. Note that each index indicates
        the state of a pin and that 1 means "HIGH" and 0 means "LOW". This
        is accomplished comtantly adjusting a list represing the state of the
        motor as described above. The stages are represended as two different
        lists, one for the stage with only one motor on 1 and the other for
        the state in which two motors are on 1


        [1,0,0,0]
        [1,1,0,0]
        [0,1,0,0]
        [0,1,1,0]
        [0,0,1,0]
        [0,0,1,1]
        [0,0,0,1]
        [1,0,0,1]

        """

        # direction input validation
        assert (direction in self.possibleDrives), "Valid directions must be strings and can be 'CW' or 'CCW'"


        # if given a negative angle, turn the opposite
        # direction for the given direction
        if degrees < 0:
            # create a dummy list containing the possible
            # drives, "CW" and "CCW", remove the current
            # and assign direction to the other, what is
            # left
            temp_keys = list(self.possibleDrives.keys())
            temp_keys.remove(direction)
            direction = temp_keys[0]
            degrees = abs(degrees)


        # assign drive to the proper list-altering
        # helper function according to the direction
        drive = self.possibleDrives[direction]


        # convesion from degrees to steps
        numSteps = int(self.stepsPerRot * (degrees / 360.))

        # HIGH LOW  LOW LOW or LOW HIGH LOW LOW depending on direction
        stage1 = [0,1,0,0] if direction == "CCW" else [1,0,0,0]
        stage2 = [1,1,0,0] # HIGH HIGH LOW LOW


        # at each step, set all the pins to states
        # correstonding to either stage 1 (one high)
        # or stage 2 (two high) then adjust the list
        # with helper functions for the next iteration
        for step in range(numSteps):

            # determine which stage to set the pins to
            stage = stage1 if step % 2 == 1 else stage2

            # set pins
            for i in range(len(self.pins)):
                self.pins[i].write(stage[i])
                time.sleep(self.sleepTime)


            # adjust pin position for next iteration
            drive(stage)


        # positive angles go "CCW", so add if going
        # counterclockwise and subtract if going
        # clockwise
        if direction == "CCW":
            newLocation = self.location + degrees
        else:
            newLocation = self.location - degrees

        # fix range of the location
        self.location = Stepper.validifyLoc(newLocation)

        return self.location

    def calibrate(self):
        """
        Resets the motor's position to zero, calibrating
        it in place.

        Args, Retuns: None
        """
        self.location = 0

    def stick(self, stick=None):
        if stick is None: stick = JoyStick(self.board)
        while True:
            x,y,b = stick.read()
            print(x,y,b)
            """
            if 0 < x < 250:
                self.turn(5)
            if 250 < x < 512:
                self.turn(-5)
            if b == 1:
                return
            """

    def __str__(self):
        return "MOTOR USING PINS" + str(self.pinLabels)

    @staticmethod
    def validifyLoc(loc):
        """
        Args:
        loc (int, float): Position given as any real number

        Returns:
        (int): valid angle

        Takes a angular location in the real numbers and returns a
        valid angular location such that loc such that 0 <= loc < 360

        NOTE: angular positions in degrees
        """

        # for float input
        loc = int(round(loc))

        # account for negative position
        while loc < 0:
            loc += 360

        # account for position greater that 360
        loc = loc % 360

        return loc

    @staticmethod
    def CCW_drive(stage):
        """
        Args:
        stage (list): stage to operate on

        Modifies GIVEN stage such that the motor is being driven
        will move counterclockwise.

        That is, the value originally
        at position stage[-1] is put at the beggining and everything
        else is shifted accordingly.

        NOTE: that this performs the operation on the given list and
        does not create a new list
        """
        stage.insert(0, stage[-1])
        stage.pop()

    @staticmethod
    def CW_drive(stage):
        """
        Args:
        stage (list): stage to operate on

        Modifies GIVEN stage such that the motor is being driven
        will move clockwise.

        That is, the value originally
        at position stage[0] is removed and put at the end.

        NOTE: that this performs the
        operation on the given list and does not create a new list
        """
        stage.append(stage[0])
        stage.pop(0)

class JoyStick():
    def __init__(self, board, pins=[4,5,8]):
        self.board = board
        self.x_pin = board.get_pin("a:" + str(pins[0]) + ":i")
        self.y_pin = board.get_pin("a:" + str(pins[1]) + ":i")
        self.b_pin = board.get_pin("d:" + str(pins[2]) + ":i")
        self.pins = [self.x_pin, self.y_pin, self.b_pin]
        self.position = [0,0,0] # (x,y,on/off)

    def read(self):
        print(self.board.analog[0].read())
        self.position = [pin.read() for pin in self.pins]
        time.sleep(1)
        return self.position






