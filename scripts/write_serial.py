import click
import serial


@click.command()
@click.argument('baud_rate', type=int)
def serial_loop(baud_rate):
    with serial.Serial('/dev/ttyACM0', baud_rate, timeout=None) as ser:
        while True:
            val = input("> ")
            ser.write(val.encode('ascii'))
            ser.flush()

if __name__ == "__main__":
    serial_loop()
