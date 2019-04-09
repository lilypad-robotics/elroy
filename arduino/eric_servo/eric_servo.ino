#include <Servo.h>

Servo myservo;

int pos = 0;    // variable to store the servo position
float current = 0;
void setup() {
	Serial.begin(9600);
	Serial.setTimeout(100);
	Serial.println("Hello world");
	myservo.attach(9);  // attaches the servo on pin 9 to the servo object
}

void loop() {
	while (Serial.available() > 0) {
		float center = Serial.parseFloat() - 208;
		float delta = center/208*45;
		float num = myservo.read() + delta;
		myservo.write(num);
	}
}
