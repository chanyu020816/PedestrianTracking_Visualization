from shapely.geometry import Polygon

HEIGHT=850
WIDTH=1750
FRAME_INTERVAL=1

REGION_DATA = [


]

REGIONS = [
    {
        'id': 0,
        "name": "往大學城方向",
        "polygon": Polygon([(600, 200), (900, 200), (900, 450), (600, 450)]),
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    },
   {
       'id': 1,
        "name": "文學館",
        "polygon": Polygon([(0, 250), (240, 200), (240, 500), (0, 550)]),  # Adjusted coordinates to right edge
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (255, 255, 225),  # Region Text Color
    },
    {
        'id': 2,
        "name": "商館二樓出口",
        "polygon": Polygon([(100, 550), (700, 500), (700, 800), (100, 800)]),
        "counts": 0,
        "dragging": False,
        "region_color": (0, 255, 0),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]