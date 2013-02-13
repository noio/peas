"""
    Module implements a checkers game with alphabeta search and input for a heuristic
"""

### IMPORTS ###
import numpy as np


### CONSTANTS ###

EMPTY = 0
WHITE = 1
BLACK = 2
MAN   = 4
KING  = 8
FREE  = 16 

NUMBERING = np.array([[ 4,  0,  3,  0,  2,  0,  1,  0],
                      [ 0,  8,  0,  7,  0,  6,  0,  5],
                      [12,  0, 11,  0, 10,  0,  9,  0],
                      [ 0, 16,  0, 15,  0, 14,  0, 13],
                      [20,  0, 19,  0, 18,  0, 17,  0],
                      [ 0, 24,  0, 23,  0, 22,  0, 21],
                      [28,  0, 27,  0, 26,  0, 25,  0],
                      [ 0, 32,  0, 31,  0, 30,  0, 29]])
                 
INVNUM = dict([(n, tuple(a[0] for a in np.nonzero(NUMBERING == n))) for n in range(1,NUMBERING.max())])


### FUNCTIONS ###

def alphabeta(game, heuristic_a, heuristic_b, max_ply=4):
    pass

### CLASSES ###

class Checkers(object):
    """ Represents the checkers game(state)
    """

    def __init__(self):
        """ Initialize the game board. """
        self.board = NUMBERING.copy()
        tiles = self.board > 0
        self.board[tiles] = EMPTY
        self.board[:3,:] = BLACK | MAN
        # self.board[3, :] = WHITE | MAN
        self.board[5:,:] = WHITE | MAN
        self.board[np.logical_not(tiles)] = FREE
        
        self.to_move = BLACK
        print NUMBERING
        print self
        for move in self.moves():
            print move

               
    def moves(self):
        """ Return a list of possible moves. """
        for n, (y,x) in INVNUM.iteritems():
            piece = self.board[(y,x)]
            if piece & self.to_move:
                if piece & MAN:
                    nextrow = y + 1 if self.to_move == BLACK else y - 1
                    if x - 1 >= 0 and self.board[nextrow, x - 1] == EMPTY:
                        yield (n, NUMBERING[nextrow, x - 1])
                    if x + 1 <= 7 and self.board[nextrow, x + 1] == EMPTY:
                        yield (n, NUMBERING[nextrow, x + 1])
                    for m in self.captures((y,x), self.to_move, self.board):
                        if len(m) > 1:
                            yield m

    
    def captures(self, (py, px), color, board, captured=[], start=None):
        """ Return a list of possible capture moves for given piece. """
        if start is None:
            start = (py, px)
        # Look for capture moves
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                # Check if the target square is on the board
                tx, ty = px + (dx + dx), py + (dy + dy)
                if (ty, tx) == start or (0 <= tx < 7 and 0 <= ty < 7 and board[ty, tx] == EMPTY):
                    # Check if the jumped square is the opposite color
                    jx, jy = px + dx, py + dy
                    if ((jy, jx) not in captured and
                        ((color == WHITE) and (board[jy, jx] & BLACK) or
                         (color == BLACK) and (board[jy, jx] & WHITE))):
                        for sequence in self.captures((ty, tx), color, board, captured + [(jy, jx)], start):
                            yield [NUMBERING[py, px]] + sequence
        yield [NUMBERING[py, px]]
                        
        
    def play(self, move):
        """ Play the given move on the board. """
        
    def game_over(self):
        """ Whether the game is over. """
        
    def evaluate(self):
        """ Returns board score. """
        
    def __str__(self):
        s = np.array([l for l in "-    wb  WB      "])
        s = s[self.board]
        if self.to_move == BLACK:
            s[0,7] = 'v'
        else:
            s[7,0] = '^'
        s = '\n'.join(' '.join(l) for l in s)
        return s
    
### PROCEDURE ###

if __name__ == '__main__':
    c = Checkers()