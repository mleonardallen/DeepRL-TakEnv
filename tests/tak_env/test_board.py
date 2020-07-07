from pytest import fixture
from tak_env import board, types
import numpy as np

def describe_add_layer():

  @fixture
  def state():
    return np.ones((1, 3, 3))

  def it_adds_layer(state):
    newState = board.add_layer(state)
    assert newState.tolist() == np.array([
      [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ]
    ]).tolist()

def describe_is_addjacent():
  def it_returns_true_if_adjacent_horizontal():
    assert board.is_adjacent((0,0), (0,1)) == True
  def it_returns_true_if_adjacent_vertical():
    assert board.is_adjacent((0,0), (1,0)) == True
  def it_is_not_adjacent_if_diagonal():
    assert board.is_adjacent((0,0), (1,1)) == False

def describe_get_top_index():
  @fixture
  def state():
    return np.array([
      # idx = 0
      [
        [1,0,1],
        [0,0,0],
        [1,0,1],
      ],
      # idx = 1
      [
        [1,0,1],
        [1,0,1],
        [1,1,1],
      ]
    ])

  def it_returns_negative_one_when_layer_needed(state):
    assert board.get_top_index(state, (0,0)) == -1
  def it_returns_zero_when_highest_layer(state):
    assert board.get_top_index(state, (1,0)) == 0
  def it_returns_index_of_first_empty_layer(state):
    assert board.get_top_index(state, (0,1)) == 1

def describe_get_size():
  @fixture
  def state():
    return np.array([
      [
        [0,0,0],
        [0,0,0],
        [0,0,0],
      ],
    ])

  def it_returns_size(state):
    assert board.get_size(state) == 3

def describe_get_height():
  @fixture
  def state():
    return np.array([
      [
        [0,0,0],
        [0,0,0],
        [0,0,0],
      ],
      [
        [0,0,0],
        [0,0,0],
        [0,0,0],
      ],
    ])

  def it_returns_height(state):
    assert board.get_height(state) == 2

def describe_get_top_layer():
  @fixture
  def state():
    return np.array([
      [
        [0,0,0],
        [0,-1,0],
        [0,0,0],
      ],
      [
        [0,1,0],
        [1,1,1],
        [0,1,0],
      ],
      [
        [2,0,2],
        [2,0,2],
        [2,0,2],
      ]
    ])
  
  def it_merges_down_top(state):
    assert board.get_top_layer(state).tolist() == np.array([
      [2,1,2],
      [1,-1,1],
      [2,1,2],
    ]).tolist()

def describe_get_open_spaces():

  def it_returns_open_spaces():
    assert board.get_open_spaces(np.array([
      [
        [0,0,0],
        [0,1,0],
        [0,0,0],
      ],
      [
        [0,1,0],
        [1,1,1],
        [0,1,0],
      ],
    ])) == [
      (0, 0),
      (0, 2),
      (2, 0),
      (2, 2)
    ]

  def it_returns_no_open_spaces():
    assert board.get_open_spaces(np.array([
      [
        [0,0,0],
        [0,1,0],
        [0,0,0],
      ],
      [
        [1,1,1],
        [1,1,1],
        [1,1,1],
      ],
    ])) == []

def describe_has_open_spaces():

  def it_returns_open_spaces():
    assert board.has_open_spaces(np.array([
      [
        [0,0,0],
        [0,1,0],
        [0,0,0],
      ],
      [
        [0,1,0],
        [1,1,1],
        [0,1,0],
      ],
    ])) == True

  def it_returns_no_open_spaces():
    assert board.has_open_spaces(np.array([
      [
        [0,0,0],
        [0,1,0],
        [0,0,0],
      ],
      [
        [1,1,1],
        [1,1,1],
        [1,1,1],
      ],
    ])) == False

def describe_get_matching_spaces():
  @fixture
  def state():
    return np.array([
      [
        [1,0,-1],
        [2,0,-2],
        [3,0,-3],
      ]
    ])

  def it_returns_matching_spaces(state):
    assert board.get_matching_spaces(
      state, 
      [types.Stone.CAPITAL.value]
    ) == [(2,0)]

  def it_returns_matching_for_negative_values(state):
    assert board.get_matching_spaces(
      state,
      [types.Stone.CAPITAL.value, -types.Stone.CAPITAL.value]
    ) == [(2,0),(2,2)]

def describe_get_pieces_at_space():
  @fixture
  def state():
    return np.array([
      [
        [0,0,0],
        [1,0,1],
        [0,0,0],
      ],
      [
        [0,2,0],
        [2,-2,2],
        [0,2,0],
      ],
      [
        [0,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ])
  
  def it_returns_top_pieces(state):
    assert board.get_pieces_at_space(state, (1,1)).tolist() == [-2, 3]
  
  def it_returns_zero_for_empty_space(state):
    assert board.get_pieces_at_space(state, (0,0)).tolist() == [0]

def describe_get_owned_spaces():
  @fixture
  def state():
    return np.array([
      [
        [1,2,3],
        [-1,-2,-3],
        [0,0,0],
      ]
    ])
  
  def it_returns_players_owned_spaces(state):
    assert board.get_owned_spaces(state, 1) == [(0,0), (0,1), (0,2)]

  def it_returns_other_players_owned_spaces(state):
    assert board.get_owned_spaces(state, -1) == [(1,0), (1,1), (1,2)]

  def it_returns_players_spaces_for_given_stone_types(state):
    assert board.get_owned_spaces(state, 1, [types.Stone.FLAT]) == [(0,0)]

def describe_remove():
  @fixture
  def state():
    return np.array([
      [
        [0,0,0],
        [1,0,1],
        [0,0,0],
      ],
      [
        [0,2,0],
        [2,-2,2],
        [0,2,0],
      ],
      [
        [0,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ])
  
  def it_removes_n_pieces(state):
    assert board.remove(state, (1,0), 2).tolist() == [
      [
        [0,0,0],
        [0,0,1],
        [0,0,0],
      ],
      [
        [0,2,0],
        [0,-2,2],
        [0,2,0],
      ],
      [
        [0,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ]

def describe_put():
  @fixture
  def state():
    return np.array([
      [
        [0,0,0],
        [1,0,1],
        [0,0,0],
      ],
      [
        [0,2,0],
        [2,-2,2],
        [0,2,0],
      ],
      [
        [0,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ])
  
  def it_puts_piece_at_space(state):
    assert board.put(state, (0,0), [1]).tolist() == [
      [
        [0,0,0],
        [1,0,1],
        [0,0,0],
      ],
      [
        [0,2,0],
        [2,-2,2],
        [0,2,0],
      ],
      [
        [1,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ]

  def it_puts_piece_at_space_in_correct_order(state):
    assert board.put(state, (0,0), [2, 1]).tolist() == [
      [
        [0,0,0],
        [1,0,1],
        [0,0,0],
      ],
      [
        [2,2,0],
        [2,-2,2],
        [0,2,0],
      ],
      [
        [1,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ]

  def it_puts_3_pieces_at_space(state):
    assert board.put(state, (0,0), [1, 1, 1]).tolist() == [
      [
        [1,0,0],
        [1,0,1],
        [0,0,0],
      ],
      [
        [1,2,0],
        [2,-2,2],
        [0,2,0],
      ],
      [
        [1,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ]

  def it_puts_flattens_standing_with_capstone(state):
    assert board.put(state, (1,1), [3]).tolist() == [
      [
        [0,0,0],
        [1,3,1],
        [0,0,0],
      ],
      [
        [0,2,0],
        [2,-1,2],
        [0,2,0],
      ],
      [
        [0,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ]

  def it_puts_pieces_at_space_and_adds_layers_if_needed(state):
    assert board.put(state, (0,0), [1, 1, 1, 1, 1]).tolist() == [
      [
        [1,0,0],
        [0,0,0],
        [0,0,0],
      ],
      [
        [1,0,0],
        [0,0,0],
        [0,0,0],
      ],
      [
        [1,0,0],
        [1,0,1],
        [0,0,0],
      ],
      [
        [1,2,0],
        [2,-2,2],
        [0,2,0],
      ],
      [
        [1,3,0],
        [3,3,3],
        [0,3,0],
      ]
    ]

def describe_move_part():

  @fixture
  def state():
    return np.array([
      [
        [0,0,0],
        [0,2,0],
        [0,0,0],
      ],
      [
        [0,-3,0],
        [1,1,0],
        [0,0,0],
      ]
    ])

  def xxx(state):
    assert board.move_part(state, (1,1), (1,2), 2).tolist() == [
      [
        [0,0,0],
        [0,0,2],
        [0,0,0],
      ],
      [
        [0,-3,0],
        [1,0,1],
        [0,0,0],
      ]
    ]

def describe_get_next_space():

  @fixture
  def board_size():
    return 3

  def it_gets_next_space_up(board_size):
    assert board.get_next_space(board_size, (1,1), types.Direction.UP.value) == (0,1)

  def it_gets_next_space_down(board_size):
    assert board.get_next_space(board_size, (1,1), types.Direction.DOWN.value) == (2,1)

  def it_gets_next_space_left(board_size):
    assert board.get_next_space(board_size, (1,1), types.Direction.LEFT.value) == (1,0)

  def it_gets_next_space_right(board_size):
    assert board.get_next_space(board_size, (1,1), types.Direction.RIGHT.value) == (1,2)

  def it_returns_none_if_invalid(board_size):
    assert board.get_next_space(board_size, (0,0), types.Direction.UP.value) == None
