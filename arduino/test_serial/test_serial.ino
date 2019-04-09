void setup() {
	Serial.begin(14400);
	Serial.println("Hello world");
	Serial.setTimeout(10);
}

float incomingByte = 0;
void loop() {
	if (Serial.available() > 0) {
	    // read the incoming byte:
	    incomingByte = Serial.parseFloat();

	    // say what you got:
	    Serial.print("I received: ");
	    Serial.println(incomingByte);
	  }
}
