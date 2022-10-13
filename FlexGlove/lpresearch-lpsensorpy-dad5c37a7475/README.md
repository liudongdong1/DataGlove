# LpSensorPy

This is a python library for ME/B2 sensors. To run the program please `pip install pyserial` package.

You could with Python 2 using the `master` branch, or Python 3 with the `python3` branch.

## Getting started

To connect to a B2 sensor via bluetooth, you need to first connect the sensor in system bluetooth manager. 

Please change `port = 'COM64'` in LpmsB2.py to the actual COM port your sensor is connected to.

You may [open the `Devices and Printers` panel](https://www.top-password.com/blog/open-the-devices-and-printers-in-windows-10/) and check the COM port of your sensor.

![Sensor Properties Windows](https://bitbucket.org/lpresearch/lpsensorpy/raw/f6d8b902cd24c7729faf965fe063f0843d37e700/images/sensor_property_window.png)