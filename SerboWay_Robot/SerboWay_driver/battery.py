import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
class Battery:
    def __init__(self, min_value=2396, max_value=3215):
        self.min_value = min_value
        self.max_value = max_value
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(self.i2c)
        self.chan = AnalogIn(self.ads, ADS.P0)
    def calculate_percentage(self, value):
        if value < self.min_value:
            return 0.0
        elif value > self.max_value:
            return 100.0
        else:
            percentage = (value - self.min_value) / (self.max_value - self.min_value) * 100
            return percentage
    def get_battery(self, sample_count=10):
        values = []
        for _ in range(sample_count):
            values.append(self.chan.value)
            time.sleep(0.1)
        avg_value = sum(values) / len(values)
        percentage = round(self.calculate_percentage(avg_value), 1)
        return percentage
    def clean(self):
        self.i2c.deinit()
    def __del__(self):
        self.clean()
