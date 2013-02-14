"""
    Module implements a checkers game with alphabeta search and input for a heuristic
"""

### IMPORTS ###
import random

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
                 
INVNUM = dict([(n, tuple(a[0] for a in np.nonzero(NUMBERING == n))) for n in range(1, NUMBERING.max() + 1)])


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
               
    def moves(self):
        """ Return a list of possible moves. """
        captures_possible = False
        pieces = []
        # Check for possible captures first:
        for n, (y,x) in INVNUM.iteritems():
            piece = self.board[y, x]
            if piece & self.to_move:
                pieces.append((n, (y, x)))
                for m in self.captures((y, x), piece, self.board):
                    if len(m) > 1:
                        captures_possible = True
                        yield m
        # Otherwise check for normal moves:
        if not captures_possible:
            for (n, (y, x)) in pieces:    
                nextrow = y + 1 if self.to_move == BLACK else y - 1
                if 0 <= nextrow < 8:
                    if x - 1 >= 0 and self.board[nextrow, x - 1] == EMPTY:
                        yield (n, NUMBERING[nextrow, x - 1])
                    if x + 1 < 8 and self.board[nextrow, x + 1] == EMPTY:
                        yield (n, NUMBERING[nextrow, x + 1])

    
    def captures(self, (py, px), piece, board, captured=[], start=None):
        """ Return a list of possible capture moves for given piece. """
        if start is None:
            start = (py, px)
        # Look for capture moves
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                dist = 1 
                while True:
                    jx, jy = px + dist * dx, py + dist * dy # Jumped square
                    # Check if piece at jx, jy:
                    if not (0 <= jx < 8 and 0 <= jy < 8):
                        break
                    if board[jy, jx] != EMPTY:
                        tx, ty = px + (dist + 1) * dx, py + (dist + 1) * dy # Target square
                        # Check if it can be captured:
                        if ((0 <= tx < 8 and 0 <= ty < 8) and
                            ((ty, tx) == start or board[ty, tx] == EMPTY) and
                            (jy, jx) not in captured and
                            ((piece & WHITE) and (board[jy, jx] & BLACK) or
                             (piece & BLACK) and (board[jy, jx] & WHITE))
                            ):
                            # Normal pieces cannot continue capturing after reaching last row
                            if not piece & KING and (piece & WHITE and ty == 0 or piece & BLACK and ty == 7):
                                yield [NUMBERING[py, px], NUMBERING[ty, tx]]
                            else:
                                for sequence in self.captures((ty, tx), piece, board, captured + [(jy, jx)], start):
                                    yield [NUMBERING[py, px]] + sequence
                        break
                    else:
                        if piece & MAN:
                            break
                    dist += 1
        yield [NUMBERING[py, px]]
                        
        
    def play(self, move):
        """ Play the given move on the board. """
        positions = [INVNUM[p] for p in move]
        (ly, lx) = positions[0]
        # Check for captures
        for (py, px) in positions[1:]:
            ydir = 1 if py > ly else -1
            xdir = 1 if px > lx else -1
            for y, x in zip(xrange(ly + ydir, py, ydir),xrange(lx + xdir, px, xdir)):
                self.board[y,x] = EMPTY
            (ly, lx) = (py, px)
        # Move the piece
        (ly, lx) = positions[0]
        (py, px) = positions[-1]
        piece = self.board[ly, lx]
        self.board[ly, lx] = EMPTY
        # Check if the piece needs to be crowned
        if piece & BLACK and py == 7 or piece & WHITE and py == 0:
            piece = piece ^ MAN | KING
        self.board[py, px] = piece

        self.to_move = WHITE if self.to_move == BLACK else BLACK
        
    def game_over(self):
        """ Whether the game is over. """
        return not list(self.moves())
        
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
    white_wins = 0
    black_wins = 0
    for i in range(100):
        c = Checkers()
        while not c.game_over():
            picked = random.choice(list(c.moves()))
            c.play(picked)
        black_win = c.to_move == WHITE
        print "BLACK wins" if black_win else "WHITE wins"
        if black_win:
            black_wins += 1
        else:
            white_wins += 1
        print c
        print black_wins, white_wins