import tak_env.board as Board
from tak_env.types import Player, Stone

def get_points(size, available_pieces, player, winner):

    # game tied if no winner
    if not winner:
        return 0

    pieces = available_pieces.get(winner)
    score = size ** 2
    score += pieces.get('pieces')
    score += pieces.get('capstones')

    # will make negative if player lost
    score = score * player * winner
    return score

def get_flat_winner(state):
    """
    If no one accomplishes a road win, you can also win by controlling the most spaces with flat stones when the game ends.
    The game ends when all spaces are covered, or when one player places his last piece. 
    This is called a "flat win" or "winning the flats."
    If the flatstone score was tied it is just a tie.

    http://cheapass.com//wp-content/uploads/2016/07/Tak-Beta-Rules.pdf
    """
    white = Board.get_owned_spaces(state, Player.WHITE.value, stones_types = [Stone.FLAT])
    black = Board.get_owned_spaces(state, Player.BLACK.value, stones_types = [Stone.FLAT])

    if len(white) > len(black):
        return Player.WHITE.value
    elif len(white) < len(black):
        return Player.BLACK.value

    return 0
