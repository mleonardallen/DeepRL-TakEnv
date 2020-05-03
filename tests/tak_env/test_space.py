from pytest import fixture
from tak_env import space, types, types
import numpy as np

def describe_can_move():

  @fixture
  def state():
    return np.array([
      [
        [1,0,0],
        [2,0,0],
        [1,0,0],
      ],
      [
        [1,0,2],
        [1,0,1],
        [1,1,2],
      ]
    ])


  def should_not_allow_move_on_standing(state):
    assert space.can_move(state, (0,2), [types.Stone.FLAT.value]) == False

  def should_allow_move_on_standing_if_capital(state):
    assert space.can_move(state, (0,2), [types.Stone.CAPITAL.value]) == True

  def should_allow_move_on_flat(state):
    assert space.can_move(state, (0,0), [types.Stone.FLAT.value]) == True

  def should_allow_move_on_empty(state):
    assert space.can_move(state, (0,1), [types.Stone.FLAT.value]) == True

  def should_account_for_top_piece(state):
    assert space.can_move(state, (1,0), [types.Stone.FLAT.value]) == False

