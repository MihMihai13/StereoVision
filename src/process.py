from PIL import Image
import numpy as np


class Process:
    # Helper class for handling image loading, conversion, resizing, and basic manipulation.
    def __init__(self, image):
        self.image = Image.open(image)
        self.image = self.image.convert('L')

    def get_info(self):
        im = self.image
        print(f"Image size: {im.size}")
        print(f"Image mode: {im.mode}")
        return im.size, im.mode

    def get_array(self):
        self.array = np.array(self.image, dtype='int64')
        return self.array

    def resize_image(self, width, height):
        new_dim = (width, height)
        self.image = self.image.resize(new_dim)
        return 

    def slice_image(self, xi, yi, size):
        # Extract a square region (slice) from the image array.
        self.slice = self.array[xi:(xi + size), yi:(yi + size)]
        return self.slice

    def show_image(self):
        self.image.show()
        return




