#include <ArduinoJson.h>
#include <Servo.h>

Servo myservo;
void setup() {
	myservo.attach(9);  // attaches the servo on pin 9 to the servo object
	Serial.begin(9600);
	while (!Serial) continue;
}

DynamicJsonDocument msg(1024);
float velocity = 0;
float damping = 0;
void loop() {
	while (!Serial.available())
		delay(50);

	auto error = deserializeJson(msg, Serial);
	if (error) {
		return;
	}
	float center = msg["mean"];
	float distance = 100 * center * center;
	if (center < 0) {
		myservo.write(myservo.read() - distance);
	} else {
		myservo.write(myservo.read() + distance);
	}

}
