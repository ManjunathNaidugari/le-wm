"""Configure the RGB camera used by all supported simulation workflows."""

def rgb_sensor(habitat_sim, settings):
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = 'rgb'
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [settings.resolution, settings.resolution]
    sensor.position = [0., settings.camera_height_m, 0.]
    sensor.hfov = settings.hfov_degrees
    return sensor
