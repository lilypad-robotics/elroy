import click
import serial


@click.command()
@click.argument('baud_rate', type=int)
def serial_loop(baud_rate):
    with serial.Serial('/dev/ttyACM0', baud_rate, timeout=None) as ser:
        while True:
            line = ser.readline()   # read a '\n' terminated line
            print("-> " + line.decode('ascii'), end='')

if __name__ == "__main__":
    serial_loop()
