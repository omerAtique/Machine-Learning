from PIL import Image
import pandas as pd
import numpy as np

imageName = 'DALL·E 2022-07-21 10.08.29 - an panda astronaut listening to music in space with the earth behind him in style of pixel art.png'

im = Image.open(imageName)

pixels = list(im.getdata())

result = []
counter = 0

for pixel in pixels:
    counter += 1
    result.append(['pixel'+ str(counter), pixel[1]])


df = pd.DataFrame(result, columns = ['Pixel', 'Pic'])

df = df.set_index('Pixel')
df = df.transpose()

df.to_csv('result.csv', index = False)
