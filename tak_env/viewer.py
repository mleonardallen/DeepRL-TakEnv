import importlib
import time
from tak_env.stone import Stone
from tak_env.board import Board
import os
import sys

class Viewer():

    def __init__(self, env, delay=0.5):

        self.block_size = 150
        self.env = env
        self.board_size = (self.env.board.size, self.env.board.size)

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

        self.width = self.block_size * self.board_size[0]

        self.bg_color = self.colors['white']
        self.pygame = None

    def init(self):
        if self.pygame:
            return

        self.pygame = importlib.import_module('pygame')
        self.pygame.init()

        image_dir = os.path.dirname(__file__) + '/images/'

        white_flat = self.pygame.image.load(image_dir + 'white.png')
        size = white_flat.get_size()
        size = (int(size[0] / 4), int(size[1] / 4))
        white_flat = self.pygame.transform.scale(white_flat, size)

        black_flat = self.pygame.image.load(image_dir + 'black.png')
        size = black_flat.get_size()
        size = (int(size[0] / 4), int(size[1] / 4))
        black_flat = self.pygame.transform.scale(black_flat, size)

        white_standing = self.pygame.image.load(image_dir +'white_standing.png')
        size = white_standing.get_size()
        size = (int(size[0] / 4), int(size[1] / 4))
        white_standing = self.pygame.transform.scale(white_standing, size)

        black_standing = self.pygame.image.load(image_dir + 'black_standing.png')
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
        self.font = self.pygame.font.Font(None, 64)
        self.font_small = self.pygame.font.SysFont("monospace", 24)

    def render(self, state):

        self.init()
        self.screen.fill(self.bg_color)
        self.draw_lines()

        for rowidx, row in enumerate(state):
            for colidx, column in enumerate(row):
                for height, stone in enumerate(column):
                    self.stone(stone, (rowidx, colidx), height)

        black = self.env.board.available_pieces.get(Board.BLACK)
        white = self.env.board.available_pieces.get(Board.WHITE)

        white_label = self.font_small.render('White:' + str(white.get('pieces')), 1, (0,0,255))
        self.screen.blit(white_label, (self.width - 120, 2))
        black_label = self.font_small.render('Black:' + str(black.get('pieces')), 1, (0,0,255))
        self.screen.blit(black_label, (self.width - 120, 24))

        if self.env.done:
            reward = self.env.reward * self.env.turn
            if reward > 0:
                text = 'White Wins!'
            elif reward < 0:
                text = 'Black Wins!'
            else:
                text = 'Tie'

            label = self.font.render(text, 1, (255,0,0))
            self.screen.blit(label, (2, 2))

        self.pygame.display.flip()
        if self.env.done:
            if sys.version_info >= (3, 0):
                time.sleep(self.delay)
            else:
                time.sleep(self.delay)
        else:
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
