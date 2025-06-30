// Arduino Sketch
void setup() {
  pinMode(13, OUTPUT);  // Use onboard LED or external LED at pin 13
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char signal = Serial.read();
    if (signal == '1') {
      digitalWrite(13, HIGH);  // Turn LED on
      delay(300);              // Wait 300ms
      digitalWrite(13, LOW);   // Turn LED off
    }
  }
}
