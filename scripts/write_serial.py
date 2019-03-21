import serial

with serial.Serial('/dev/ttyACM0', 9600, timeout=None) as ser:
	while True:
		val = input("> ")
		ser.write(val.encode('ascii'))
