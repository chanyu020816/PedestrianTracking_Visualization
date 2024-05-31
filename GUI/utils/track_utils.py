import cv2
import numpy as np
import hashlib
import colorsys

def get_four_direction(start, end):
    """
    :param start: (x, y)
    :param end: (x, y)
    :return: direction
    """
    (x_start, y_start), (x_end, y_end) = start, end
    direction = [(x_start <= x_end), (y_start <= y_end)]

    if x_start < 0 or y_start < 0:
        dire = -1
    if direction[0] and direction[1]:
        dire = 0
    elif not direction[0] and direction[1]:
        dire = 1
    elif direction[0] and not direction[1]:
        dire = 3
    else:
        dire = 2
    return [direction, dire]

def get_horizon_direction(start, end) -> str:
    return "right" if start[0] <= end[0] else "left"

def get_vertical_direction(start, end) -> str:
    return "top" if start[1] <= end[1] else "down"

def get_eight_direction(start, end) -> int:
    quan = get_four_direction(start, end)
    slope = abs((start[1] - end[1]) / (start[0] - end[0] + 1e-7))
    dire = (1 + 2 * quan[1]) + ((quan[0][0] ^ quan[0][1]) ^ (slope >= 1))

    return dire % 8

def id_to_color(id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:

    # Hash the ID to get a consistent unique value
    hash_object = hashlib.sha256(str(id).encode())
    hash_digest = hash_object.hexdigest()

    # Convert the first few characters of the hash to an integer
    # and map it to a value between 0 and 1 for the hue
    hue = int(hash_digest[:8], 16) / 0xffffffff

    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
    rgb_255 = tuple(int(component * 255) for component in rgb)
    hex_color = '#%02x%02x%02x' % rgb_255
    # Strip the '#' character and convert the string to RGB integers
    rgb = tuple(int(hex_color.strip('#')[i:i + 2], 16) for i in (0, 2, 4))

    # Convert RGB to BGR for OpenCV
    bgr = rgb[::-1]

    return bgr

if __name__ == '__main__':
    print(get_eight_direction((20, 20), (30, 20)))
