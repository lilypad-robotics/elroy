#include <Servo.h>

Servo myservo;

int pos = 0;    // variable to store the servo position
void setup() {
	Serial.begin(9600);
	Serial.println("Hello world");
	myservo.attach(9);  // attaches the servo on pin 9 to the servo object
}

void loop() {
	while (Serial.available() > 0) {
		float num = Serial.parseFloat();
		myservo.write(num);
		Serial.print("Moving to ");
		Serial.print(num);
		Serial.println(" degrees");
	}
}
