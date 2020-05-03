import importlib
import time
from tak_env.types import Stone, Player
import os
import sys
import numpy as np
from operator import itemgetter


class Viewer():

    def __init__(self, env, delay=0.5, block_size=150):
     
        pygame = importlib.import_module('pygame')
        pygame.init()

        board_width = env.board_size * block_size
        screen = pygame.display.set_mode([board_width, board_width])

        self.config = {
            'block_size': block_size,
            'delay': delay,
            'env': env,
            'pygame': pygame,
            'screen': screen,
        }

    def render(self, state):

        env, pygame, screen, block_size, delay = itemgetter(
            'env', 'pygame', 'screen', 'block_size', 'delay'
        )(self.config)

        screen.fill((0,0,0)) # erase screen 
        draw_lines(pygame, screen, env.board_size, block_size)
        
        board_height = state.shape[0]
        for (idx, *space), value in sorted(np.ndenumerate(state), reverse=True):
            offset = board_height - idx
            draw_stone(pygame, screen, block_size, offset, space, value)

        if env.done:
            reward = env.reward * env.turn
            if reward > 0:
                print('White Wins!')
            elif reward < 0:
                print('Black Wins!')
            else:
                print('Tie')

        pygame.display.flip()
        wait(env.done, delay)

def wait(done, delay):
    if done:
        if sys.version_info >= (3, 0):
            input("Press Enter key to continue...")
        else:
            raw_input("Press Enter key to continue...")
    else:
        time.sleep(delay)

def draw_stone(pygame, screen, block_size, height, space, value):

    if value == 0:
        return

    image = get_image(pygame, value)
    image_width, image_height = image.get_size()

    row, col = space
    stack_offset = height * 10

    posx = col * block_size + (block_size / 2) - image_width / 2
    posy = row * block_size + block_size - image_height - stack_offset

    screen.blit(image,(posx,posy))

def draw_lines(pygame, screen, board_size, block_size):
    color = (255,255,255)
    length = board_size * block_size
    for i in range(board_size):
        offset = i * block_size
        pygame.draw.line(screen, color, (offset, length), (offset, 0))
        pygame.draw.line(screen, color, (length, offset), (0, offset))

def get_image(pygame, piece):
    return {
        Stone.FLAT.value: get_image_from_filename(pygame, 'white.png'),
        Stone.STANDING.value: get_image_from_filename(pygame, 'white_standing.png'),
        Stone.FLAT.value * -1: get_image_from_filename(pygame, 'black.png'),
        Stone.STANDING.value * -1: get_image_from_filename(pygame, 'black_standing.png'),
    }[piece]

def get_image_from_filename(pygame, filename):
    image_dir = os.path.dirname(__file__) + '/images/'
    image = pygame.image.load(image_dir + filename)
    size = image.get_size()
    size = (int(size[0] / 4), int(size[1] / 4))
    return pygame.transform.scale(image, size)