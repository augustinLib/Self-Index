import os


def devices():
    return [
        device.strip()
        for device in os.environ.get("DEVICES", "cuda:0").split(",")
        if device.strip()
    ]


def compute_device():
    return devices()[0]
