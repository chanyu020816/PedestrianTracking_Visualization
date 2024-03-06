import base64
from os.path import basename

def pic2str(file):
    pic = open(file, 'rb')
    name = basename(file)
    content = '{} = {}\n'.format(name, base64.b64encode(pic.read()))
    pic.close()

    with open('ConvertedImages.py', 'a') as f:
        f.write(content)


if __name__ == '__main__':
    pic2str("./images/")
