import numpy as np

from PIL import Image, ImageDraw
from random import random, randint, seed
from colorsys import hls_to_rgb
from numba import jit, prange
from typing import Tuple, Optional

from bezier import BezierCurve


class App:
    """Класс для генерации векторных композиций.

    Использует векторные поля для создания generative art с применением кривых Безье.
    Результат рендерится в изображение с помощью Pillow.

    Attributes:
        resolution (int): Размер ячейки сетки в пикселях.
        max_velocity (float): Максимальная длина вектора.
        columns (int): Количество столбцов в сетке.
        rows (int): Количество строк в сетке.
        width (int): Ширина изображения в пикселях.
        height (int): Высота изображения в пикселях.
        radius (float): Радиус влияния векторов.
        image (Image): Объект изображения Pillow.
        draw (ImageDraw): Объект для рисования на изображении.
        grid (np.ndarray): Трехмерный массив векторов (columns, rows, 2).
    """

    def fill_grid(self) -> None:
        """Заполняет сетку случайными векторами."""
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
        self,
        size: Tuple[int, int] = (32, 18),
        resolution: int = 50,
        radius: float = 0.1,
        max_velocity: float = 10,
        color_seed: Optional[int] = None,
        line_seed: Optional[int] = None,
        vectors_seed: Optional[int] = None
    ) -> None:
        """Инициализирует генератор композиций.

        Args:
            size (Tuple[int, int]): Размер сетки в ячейках (columns, rows).
            resolution (int): Размер ячейки в пикселях.
            radius (float): Радиус влияния векторов.
            max_velocity (float): Максимальная длина вектора.
            color_seed (Optional[int]): Seed для генерации цветов.
            line_seed (Optional[int]): Seed для генерации линий.
            vectors_seed (Optional[int]): Seed для генерации векторного поля.
        """
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

    def save(self, file_name: Optional[str] = None) -> None:
        """Сохраняет и/или отображает изображение.

        Args:
            file_name (Optional[str]): Если указан, сохраняет изображение в файл.
                      Если None, только отображает изображение.
        """
        if file_name is not None:
            self.image.save(file_name)
        self.image.show()

    @staticmethod
    def get_random_color(rng: Optional[int] = None) -> Tuple[int, int, int]:
        """Генерирует случайный цвет в RGB.

        Args:
            rng (Optional[str]): Seed для воспроизводимости цветов.

        Returns:
            Кортеж (R, G, B) с значениями 0-255.
        """
        if rng is not None:
            seed(rng)
        h, l, s = randint(0, 360), 0.6, 0.7
        r, g, b = hls_to_rgb(h / 360, l, s)
        return int(255 * r), int(255 * g), int(255 * b)

    @staticmethod
    @jit(fastmath=True)
    def next_point(
        x0: float,
        y0: float,
        columns: int,
        rows: int,
        grid: np.ndarray,
        resolution: int,
        radius: float
    ) -> Tuple[float, float]:
        """Вычисляет следующую точку на основе векторного поля.

        Args:
            x0 (float): x координата. 
            y0 (float): y координата.
            columns (int): Число колонок в сетке.  
            rows (int): Число строк в сетке.
            grid (np.ndarray): Векторное поле.
            resolution (int): Размер ячейки.
            radius (float): Радиус влияния.

        Returns:
            Новые координаты (x1, y1).
        """
        x1, y1 = x0, y0

        for x in prange(columns):
            for y in prange(rows):
                if ((x * resolution - x0) ** 2 + (y * resolution - y0) ** 2) <= (radius ** 2):
                    dx, dy = grid[x, y]
                    x1 += dx
                    y1 += dy

        return x1, y1

    def draw_line(
        self,
        x_start: float,
        y_start: float,
        steps: int,
        color: Tuple[int, int, int],
        width: int
    ) -> None:
        """Рисует линию с использованием кривой Безье.

        Args:
            x_start (float): Начальная x координата. 
            y_start (float): Начальная y координата.
            steps (int): Количество шагов построения.
            color (Tuple[int, int, int]): Цвет линии в RGB.
            width (int): Толщина линии.
        """
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

        for i in prange(len(points) - 1):
            start, end = points[i], points[i + 1]
            self.draw.line([*start, *end], fill=color, width=width)

    def generate(self, number_of_lines: int = 40) -> None:
        """Генерирует композицию из линий.

        Args:
            number_of_lines (int): Количество генерируемых линий.
        """
        for _ in prange(number_of_lines):
            if self.line_seed is not None:
                seed(self.line_seed)
            x, y = random() * self.width, random() * self.height
            color = self.get_random_color(self.color_seed)
            self.draw_line(x, y, 50, color, randint(3, 8))
