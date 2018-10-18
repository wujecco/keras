#import matplotlib.image as mpimg

#img = mpimg.imread('.\sampleImage\000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png')
#print(img.shape, img.dtype)

from PIL import Image
im = Image.open('test.jpg')
im.rotate(45).show()