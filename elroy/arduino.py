import json
import time
import serial

class Serial(object):

    def __init__(self, baud_rate=9600):
        self.baud_rate = baud_rate
        self.conn = serial.Serial('/dev/ttyACM0', self.baud_rate, timeout=None)
        self.iter = 0

    def write(self, coords):
        if self.iter % 2 == 0:
            message = {
                'mean': coords[0] - 0.5
            }
            self.conn.write(json.dumps(message).encode('ascii'))
            self.conn.flush()
        self.iter += 1

    def close(self):
        self.conn.close()
