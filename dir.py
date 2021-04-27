import os

images = os.listdir('./images/bales/')

for index in range(3,10):
    print(index, images[index])

for count, value in enumerate(images):
    print(count, value)    