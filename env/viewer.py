import importlib
import time
from env.stone import Stone
import os
import getch

class Viewer():

    def __init__(self, board_size=3, delay=1):

        self.block_size = 150
        self.board_size = (board_size, board_size)

        self.colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'orange': (255, 128, 0)
        }

        self.delay = delay

        self.size = (
            (self.board_size[0]) * self.block_size,
            (self.board_size[1]) * self.block_size
        )

        self.bg_color = self.colors['white']
        self.pygame = None

    def init(self):
        if self.pygame:
            return

        self.pygame = importlib.import_module('pygame')
        self.pygame.init()

        white_flat = self.pygame.image.load('images/white.png')
        size = white_flat.get_size()
        size = (int(size[0] / 4), int(size[1] / 4))
        white_flat = self.pygame.transform.scale(white_flat, size)

        black_flat = self.pygame.image.load('images/black.png')
        size = black_flat.get_size()
        size = (int(size[0] / 4), int(size[1] / 4))
        black_flat = self.pygame.transform.scale(black_flat, size)

        white_standing = self.pygame.image.load('images/white_standing.png')
        size = white_standing.get_size()
        size = (int(size[0] / 4), int(size[1] / 4))
        white_standing = self.pygame.transform.scale(white_standing, size)

        black_standing = self.pygame.image.load('images/black_standing.png')
        size = black_standing.get_size()
        size = (int(size[0] / 4), int(size[1] / 4))
        black_standing = self.pygame.transform.scale(black_standing, size)

        self.images = {
            'WHITE': {
                'FLAT': white_flat,
                'STANDING': white_standing
            },
            'BLACK': {
                'FLAT': black_flat,
                'STANDING': black_standing
            }
        }


        self.screen = self.pygame.display.set_mode(self.size)
        self.font = self.pygame.font.Font(None, 128)

    def render(self, state):

        self.init()
        self.screen.fill(self.bg_color)
        self.draw_lines()

        for rowidx, row in enumerate(state):
            for colidx, column in enumerate(row):
                for height, stone in enumerate(column):
                    self.stone(stone, (rowidx, colidx), height)

        self.pygame.display.flip()
        time.sleep(self.delay)


    def stone(self, value, position, height):

        if value == 0:
            return

        stone = Stone(abs(value))
        key = 'WHITE' if value > 0 else 'BLACK'
        image = self.images.get(key).get(stone.name)

        image_width, image_height = image.get_size()

        row, col = position
        stack_offset = height * 10

        posx = col * self.block_size + (self.block_size / 2) - image_width / 2
        posy = row * self.block_size + self.block_size - image_height - stack_offset

        self.screen.blit(image,(posx,posy))

    def draw_lines(self):
        for i in range(self.board_size[0]):
            self.pygame.draw.line(
                self.screen,
                (0,0,0),
                (i*self.block_size, self.size[0]),
                (i*self.block_size,0)
            )

        for i in range(self.board_size[1]):
            self.pygame.draw.line(
                self.screen,
                self.colors['black'],
                (self.size[1], i*self.block_size),
                (0, i*self.block_size)
            )
