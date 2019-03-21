import serial

with serial.Serial('/dev/ttyACM0', 9600, timeout=None) as ser:
	while True:
		line = ser.readline()   # read a '\n' terminated line
		print("-> " + line.decode('ascii'), end='')
