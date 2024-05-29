import cv2
import numpy as np
import hashlib
import colorsys

horiz_grid = [10, 11, 12, 13]
verti_grid = [21, 22, 23, 24]
six_grid = [7, 8, 9]
eight_grid = [14, 15, 16]

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
    """
    :return: 1 -> left , 0 -> right
    """
    return int(start[0] <= end[0])

def get_vertical_direction(start, end) -> str:
    """
    :return: 1 -> down, 0 -> up
    """
    return int(start[1] <= end[1])

def get_eight_direction(start, end) -> int:
    quan = get_four_direction(start, end)
    slope = abs((start[1] - end[1]) / (start[0] - end[0] + 1e-7))
    dire = (1 + 2 * quan[1]) + ((quan[0][0] ^ quan[0][1]) ^ (slope >= 1))

    return dire % 8

def get_school_direction(start: int, end: int, grid_size = 200, hori_grid_num = 7) -> int:
    assert start[0] != -1

    grid = start[0] // grid_size + hori_grid_num * (start[1] // 200)

    if grid in eight_grid:
        dire = get_eight_direction(start=start, end=end)
        if dire in [2, 3]:
            return 0
        elif dire in [4, 5, 6]:
            return 2
        elif dire in [7, 0, 1]:
            return 3
    elif grid in six_grid:
        dire = get_eight_direction(start=start, end=end)
        if dire in [2, 3]:
            return 0
        elif dire == 7:
            return 1
        elif dire in [4, 5, 6]:
            return 2
        elif dire in [0, 1]:
            return 3
    elif grid in horiz_grid:
        dire = get_horizon_direction(start=start, end=end)
        if dire == 0:
            return 2
        else:
            return 3
    elif grid in verti_grid:
        dire = get_vertical_direction(start=start, end=end)
        if dire == 0:
            return 1
        else:
            return 0
    else:
        return -1


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
