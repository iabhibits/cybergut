# Apply random distortion.
# Original Author: Marcus D. Bloice <https://github.com/mdbloice>
# Modified by: Dhruv Patel
# Runs only on Python 3.x
# Licensed under the terms of the MIT Licence.

from numpy import random
from PIL import Image

class Distort:
    """A callable class. See __call__ for more information"""
    
    def __init__(self, grid_width, grid_height, magnitude):
        """
        To choose good values experiment with different paramenters.
        
        :param grid_width int: How many columns?
        :param grid_height int: How many rows?
        :param magnitude int: How much to distort?
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude
        
    def __call__(self, *images):
        """
        Distorts single image.
        
        :param images List: List of Pillow images. Each image
         is distorted using same arguments.
        :return: transformed images.
        """
        
        w, h = images[0].size
        
        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = w // horizontal_tiles
        height_of_square = h // vertical_tiles
        
        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))
        
        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                tmp = [horizontal_tile * width_of_square,
                       vertical_tile * height_of_square,
                       horizontal_tile * width_of_square,
                       vertical_tile * height_of_square]
                if horizontal_tile == horizontal_tiles - 1:
                    tmp[2] += width_of_last_square
                else:
                    tmp[2] += width_of_square
                if vertical_tile == vertical_tiles -1:
                    tmp[3] += height_of_last_square
                else:
                    tmp[3] += height_of_square
                dimensions.append(tmp)
        
        #indices of last_*
        last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_row = list(range(horizontal_tiles*vertical_tiles - horizontal_tiles, 
                              horizontal_tiles*vertical_tiles))
        polygons = [[x1,y1, x1,y2, x2,y2, x2,y1] for x1,y1, x2,y2 in dimensions]
        
        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])
        
        for a, b, c, d in polygon_indices:
            dx = random.randint(-self.magnitude, self.magnitude)
            dy = random.randint(-self.magnitude, self.magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

            generated_mesh = []
            for i in range(len(dimensions)):
                generated_mesh.append([dimensions[i], polygons[i]])

        return [im.transform(im.size, Image.MESH, generated_mesh, 
                             resample=Image.BICUBIC) for im in images]
