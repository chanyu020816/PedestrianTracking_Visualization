from PIL import ImageTk, Image
from base64 import b64decode
from io import BytesIO
from Base64Image import PedestrianTrackingPageBG

byte_data = b64decode(PedestrianTrackingPageBG)
bg_image_data = BytesIO(byte_data)
print(bg_image_data)
bg_frame = Image.open(bg_image_data)