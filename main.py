import numpy as np

from PIL import Image, ImageDraw
from random import random, randint, seed
from time import time
from colorsys import hls_to_rgb
from numba import jit, prange

from bezier import BezierCurve


class App:
    def fill_grid(self):
        if self.vector_seed is not None:
            seed(self.vector_seed)
        for x in prange(self.columns):
            for y in prange(self.rows):
                angle = 2 * np.pi * random()
                length = self.max_velocity * random()

                dx = length * np.cos(angle) + self.max_velocity / 4
                dy = length * np.sin(angle) + self.max_velocity / 15

                self.grid[x, y] = dx, dy

    def __init__(
            self, size=(32, 18), resolution=50, radius=0.1, max_velocity=10,
            color_seed=None, line_seed=None, vectors_seed=None
    ):
        self.resolution = resolution
        self.max_velocity = max_velocity

        self.columns, self.rows = size

        self.color_seed, self.line_seed, self.vector_seed = color_seed, line_seed, vectors_seed

        self.width = self.columns * self.resolution
        self.height = self.rows * self.resolution

        self.radius = radius * max(self.width, self.height)

        self.image = Image.new("RGB", (self.width, self.height), 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.grid = np.zeros((self.columns, self.rows, 2), dtype=np.float32)
        self.fill_grid()

    def save(self, file_name=None):
        if file_name is not None:
            self.image.save(file_name)
        self.image.show()

    @staticmethod
    def get_random_color(rng):
        if rng is not None:
            seed(rng)
        h, l, s = randint(0, 360), 0.6, 0.7
        r, g, b = hls_to_rgb(h / 360, l, s)
        return int(255 * r), int(255 * g), int(255 * b)

    @staticmethod
    @jit(fastmath=True)
    def next_point(x0, y0, columns, rows, grid, resolution, radius):
        x1, y1 = x0, y0

        for x in prange(columns):
            for y in prange(rows):
                if ((x * resolution - x0) ** 2 + (y * resolution - y0) ** 2) <= (radius ** 2):
                    dx, dy = grid[x, y]
                    x1 += dx
                    y1 += dy

        return x1, y1

    def draw_line(self, x_start, y_start, steps, color, width):
        x0, y0 = x_start, y_start

        points = [(x0, y0)]

        for _ in prange(steps):
            x1, y1 = self.next_point(x0, y0, self.columns, self.rows,
                                     self.grid, self.resolution, self.radius)
            points.append((x1, y1))
            x0, y0 = x1, y1

        bezier = BezierCurve(points)

        t_list = np.linspace(0, 1, max(self.width, self.height) // 2)
        points = []
        for t in t_list:
            x, y = bezier.curve(t)
            points.append((x, y))

        for _1 in prange(len(points) - 1):
            start, end = points[_1], points[_1 + 1]
            self.draw.line([*start, *end], fill=color, width=width)

    def generate(self, number_of_line=40):
        for _ in prange(number_of_line):
            if self.line_seed is not None:
                seed(self.line_seed)
            x, y = random() * self.width, random() * self.height
            color = self.get_random_color(self.color_seed)
            self.draw_line(x, y, 50, color, randint(3, 8))


if __name__ == '__main__':
    start_time = time()

    app = App(
        size=(96, 50), resolution=40, radius=0.05, max_velocity=5
    )
    app.generate(100)
    app.save(file_name='output.png')

    print(f'Time: {time() - start_time} seconds')
