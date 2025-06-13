//TCRT5000
int sensorPin = A0;    // 적외선 센서의 신호핀 연결
int ledPin = 2;      // LED 출력핀
int sensorValue = 0; 
void setup() {
  // declare the ledPin as an OUTPUT:
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}
void loop() {
  sensorValue = analogRead(sensorPin);
  Serial.println(sensorValue);
  //delay(50);
  if(sensorValue>1000)
  {
    digitalWrite(ledPin, LOW);
  }
  else
    digitalWrite(ledPin, HIGH);
}