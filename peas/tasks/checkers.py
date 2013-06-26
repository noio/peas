#!/usr/bin/env python
"""
    Module implements a checkers game with alphabeta search and input for a heuristic
"""

### IMPORTS ###
import random
import copy
import time
import sys

from collections import defaultdict

import numpy as np

### SHORTCUTS ###
inf = float('inf')


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

CENTER = [(2,2), (2,4), (3,3), (3,5), (4,2), (4,4), (5,3), (5,5)]
EDGE = [(0,0), (0,2), (0,4), (0,6), (1,7), (2,0), (3,7), (4,0), (5,7), (6,0), (7,1), (7,3), (7,5), (7,7)]
SAFEEDGE = [(0,6), (1,7), (6,0), (7,1)]
                 
INVNUM = dict([(n, tuple(a[0] for a in np.nonzero(NUMBERING == n))) for n in range(1, NUMBERING.max() + 1)])

### EXCEPTIONS

class IllegalMoveError(Exception):
    pass

### FUNCTIONS ###
def alphabeta(node, heuristic, player_max=True, depth=4, alpha=-inf, beta=inf, killer_moves=defaultdict(set), num_evals=[0]):
    """ Performs alphabeta search.
        From wikipedia pseudocode.
    """
    if depth == 0 or node.game_over():
        num_evals[0] += 1
        value = heuristic(node)
        # print "[%s] %.3f" % ("+" if player_max else "-", value),
        return value
    pmx = not player_max

    killers = killer_moves[node.turn+1]
    moves = node.all_moves()
    m = []
    for move in moves:
        if move in killers:
            m.insert(0, move)
        else:
            m.append(move)
    moves = m
    
    if player_max:
        for move in moves:
            newnode = node.copy_and_play(move)
            alpha = max(alpha, alphabeta(newnode, heuristic, pmx, depth-1, alpha, beta, killer_moves, num_evals))
            if beta <= alpha:
                if len(killers) > 4:
                    killers.pop()
                killers.add(move)
                break # Beta cutoff
        return alpha
    else:
        for move in moves:
            newnode = node.copy_and_play(move)
            beta = min(beta, alphabeta(newnode, heuristic, pmx, depth-1, alpha, beta, killer_moves, num_evals))     
            if beta <= alpha:
                if len(killers) > 4:
                    killers.pop()
                killers.add(move)
                break # Alpha cutoff
        return beta

def gamefitness(game):
    """ Returns the fitness of
        the black player. (according to {gauci2008case}) """
    counts = np.bincount(game.board.flat)
    return (100 + 2 * counts[BLACK|MAN] + 3 * counts[BLACK|KING] + 
            2 * (12 - counts[WHITE|MAN]) + 3 * (12 - counts[WHITE|KING]))

### CLASSES ###

class CheckersTask(object):
    """ Represents a checkers game played by an evolved phenotype against
        a fixed opponent.
    """
    def __init__(self, search_depth=4, 
                       opponent='simplech', 
                       opponent_search_depth=4, 
                       opponent_handicap=0.0, 
                       minefield=False, 
                       fly_kings=False, 
                       win_to_solve=3):
        self.search_depth = search_depth
        self.opponent_search_depth = opponent_search_depth
        self.opponent_handicap = opponent_handicap
        self.opponent = opponent
        self.win_to_solve = win_to_solve
        self.minefield = minefield
        self.fly_kings = fly_kings

    def evaluate(self, network):
        # Setup
        game = Checkers(minefield=self.minefield, fly_kings=self.fly_kings)
        player = HeuristicOpponent(NetworkHeuristic(network), search_depth=self.search_depth)
        if self.opponent == 'simplech':
            opponent = HeuristicOpponent(SimpleHeuristic(), search_depth=self.opponent_search_depth, handicap=self.opponent_handicap)
        elif self.opponent == 'counter':
            opponent = HeuristicOpponent(PieceCounter(), search_depth=self.opponent_search_depth, handicap=self.opponent_handicap)
        elif self.opponent == 'random':
            opponent = RandomOpponent()
        # Play the game
        fitness = []
        current, next = player, opponent
        i = 0
        print "Running checkers game between %s - %s " % (player, opponent)
        while not game.game_over():
            i += 1
            move = current.pickmove(game)
            game.play(move)
            current, next = next, current
            fitness.append(gamefitness(game))
            sys.stdout.write('.')
            sys.stdout.flush()

        if ((game.board & WHITE).sum() == 0 or (game.board & BLACK).sum() == 0) and game.winner() == 0:
            raise Exception("Game can't be a draw when one side has no pieces.")
        print
        print game
        # Fitness over last 100 episodes
        fitness.extend([gamefitness(game)] * (100 - len(fitness)))
        fitness = fitness[-100:]
        score = sum(fitness)
        won = game.winner() >= 1.0
        draw = game.winner() == 0.0
        if won:
            score += 30000
        print "\nGame finished in %d turns. Winner: %s. Score: %d. Final state fitness: %d." % (i,game.winner(), score, fitness[-1])
        return {'fitness':score, 'won': won, 'draw': draw, 'turns': i, '_move_history': game.history[:]}

    def play_against(self, network, user_side=WHITE, history=None):
        # Setup
        game = Checkers(minefield=self.minefield, fly_kings=self.fly_kings)
        # self.search_depth = 0
        player = HeuristicOpponent(NetworkHeuristic(network), search_depth=self.search_depth)        
        player.play_against(game, user_side=user_side, history=history)

    def solve(self, network):
        o = self.opponent_handicap
        self.opponent_handicap = 0.05
        for _ in range(self.win_to_solve):
            if not self.evaluate(network)['won']:
                self.opponent_handicap = o
                return False
        self.opponent_handicap = o
        return True

    def visualize(self, network, filename):
        import matplotlib.pyplot as plt
        output = NUMBERING.copy() * 0.0
        for y in range(8):
            for x in range(8):
                inpt = NUMBERING.copy() * 0
                inpt[y,x] = 1
                value = network.feed(inpt, add_bias=False, propagate=2)[-1]
                output[y,x] = value
        plt.imshow(output, vmin=-1, vmax=1, interpolation='nearest', extent=[0,8,0,8], cmap='RdYlGn')
        plt.grid(zorder=2)
        plt.savefig(filename)
        print filename
        plt.close()
            
        
class HeuristicOpponent(object):
    """ Opponent that utilizes a heuristic combined with alphabeta search
        to decide on a move.
    """
    def __init__(self, heuristic, search_depth=4, handicap=0.0):
        self.search_depth = search_depth
        self.heuristic = heuristic
        self.handicap = handicap
    
    def pickmove(self, board, verbose=False):
        player_max = (board.to_move == BLACK)
        if verbose:
            print "Picking move for player %s" % ("MAX" if player_max else "MIN")
        bestmove = None
        secondbest = None
        bestval = -inf if player_max else inf
        moves = list(board.all_moves())
        # If there is only one possible move, don't search, just move.
        if len(moves) == 1:
            if verbose: print "0 evals."
            return moves[0]
        evals = [0]
        killer_moves = defaultdict(set)
        for move in moves:
            val = alphabeta(board.copy_and_play(move), self.heuristic.evaluate, 
                depth=self.search_depth-1, player_max=not player_max, killer_moves=killer_moves, num_evals=evals)
            if verbose: 
                print board.copy_and_play(move)
                print "Value: %.3f" % val
            if (player_max and val > bestval) or (not player_max and val < bestval):
                bestval = val
                secondbest = bestmove
                bestmove = move
        # Pick second best move
        if verbose: 
            print "%d evals. value: %.2f" % (evals[0], bestval)
        if secondbest is not None and self.handicap > 0 and random.random() < self.handicap:
            return secondbest
        return bestmove

    def play_against(self, game=None, user_side=WHITE, history=None):
        if game is None:
            game = Checkers()
        
        print "Playing against %s" % self

        auto = HeuristicOpponent(SimpleHeuristic(), search_depth=4)
        full_auto = False
        on_history = True if history else False
        fitness = []
        i = 0
        while not game.game_over():
            # Computer plays first if user is white.
            if i > 0 or user_side == WHITE:
                move = self.pickmove(game, verbose=False)
                historical = history.pop(0) if history else None
                print move
                game.play(move)
                if on_history and self.handicap == 0 and move != historical:
                    raise Exception("Played (%s) different move from history (%s)." % (move, historical))

            if game.game_over():
                break
            
            fitness.append(gamefitness(game))
            print game
            moved = False
            while not moved:
                historical = history.pop(0) if history else None
                best = auto.pickmove(game)
                print "Enter move (h: %s, a: %s):" % (historical, best),
                try:
                    user_input = raw_input()
                    if user_input == 'f':
                        full_auto = True
                    if user_input == 'h' or (on_history and user_input == ''):
                        move = historical
                    elif full_auto or user_input == 'a' or (not on_history and user_input == ''):
                        move = best
                    elif user_input == 'q':
                        move = None
                    elif ' ' in user_input:
                        move = tuple(int(i) for i in user_input.split(' '))
                    else:
                        move = tuple(int(i) for i in user_input.split('-'))
                    if move is not None:
                        game.play(move)
                    moved = True
                    if move != historical:
                        on_history = False
                except (IllegalMoveError, ValueError):
                    print "Illegal move"
            fitness.append(gamefitness(game))
            print fitness
            if move is None:
                break
            print game
            time.sleep(1.0)
            i += 1
        
        fitness.extend([gamefitness(game)] * (100 - len(fitness)))
        fitness = fitness[-100:]
        print fitness
        
        print game

        won = game.winner() >= 1.0
        print "\nGame finished in %d turns. Winner: %s." % (i,game.winner())

    def __str__(self):
        return '%s with %s (lookahead: %d, handicap: %.2f)' % (self.__class__.__name__,
            self.heuristic.__class__.__name__, self.search_depth, self.handicap)
        
class SimpleHeuristic(object):
    """ Simple piece/position counting heuristic, adapted from simplech
    """
    def evaluate(self, game):
        if game.game_over():
            return 5000 * game.winner()
        board = game.board
        counts = np.bincount(board.flat)

        turn = 2;     # color to move gets +turn
        brv = 3;      # multiplier for back rank
        kcv = 5;      # multiplier for kings in center
        mcv = 1;      # multiplier for men in center
        mev = 1;      # multiplier for men on edge
        kev = 5;      # multiplier for kings on edge
        cramp = 5;    # multiplier for cramp
        opening = -2; # multipliers for tempo
        midgame = -1;
        endgame = 2;
        intactdoublecorner = 3;

        nwm = counts[WHITE|MAN]
        nwk = counts[WHITE|KING]
        nbm = counts[BLACK|MAN]
        nbk = counts[BLACK|KING]

        vb = (100 * nbm + 130 * nbk)
        vw = (100 * nwm + 130 * nwk)
        
        val = 0
        
        if (vb + vw) > 0:        
            val = (vb - vw) + (250 * (vb-vw))/(vb+vw); #favor exchanges if in material plus


        nm = nwm + nbm
        nk = nwk + nbk

        val += turn if game.to_move == BLACK else -turn

        if board[4][0] == (BLACK|MAN) and board[5][1] == (WHITE|MAN):
            val += cramp
        if board[3][7] == (WHITE|MAN) and board[2][6] == (BLACK|MAN):
            val -= cramp

        # Back rank guard
        code = 0
        if (board[0][0] & MAN): code += 1
        if (board[0][2] & MAN): code += 2
        if (board[0][4] & MAN): code += 4
        if (board[0][6] & MAN): code += 8
        if code == 1:
            backrankb = -1
        elif code in (0, 3, 9):
            backrankb = 0
        elif code in (2, 4, 5, 7, 8):
            backrankb = 1
        elif code in (6, 12, 13):
            backrankb = 2
        elif code == 11:
            backrankb = 4
        elif code == 10:
            backrankb = 7
        elif code == 15:
            backrankb = 8
        elif code == 14:
            backrankb = 9
        
        code = 0
        if (board[7][1] & MAN): code += 8
        if (board[7][3] & MAN): code += 4
        if (board[7][5] & MAN): code += 2
        if (board[7][7] & MAN): code += 1

        if code == 1:
            backrankw = -1
        elif code in (0, 3, 9):
            backrankw = 0
        elif code in (2, 4, 5, 7, 8):
            backrankw = 1
        elif code in (6, 12, 13):
            backrankw = 2
        elif code == 11:
            backrankw = 4
        elif code == 10:
            backrankw = 7
        elif code == 15:
            backrankw = 8
        elif code == 14:
            backrankw = 9
        
        val += brv * (backrankb - backrankw)

        if board[0][6] == BLACK|MAN and (board[1][5] == BLACK|MAN or board[1][7] == BLACK|MAN):
            val += intactdoublecorner
        if board[7][1] == WHITE|MAN and (board[6][0] == WHITE|MAN or board[6][2] == WHITE|MAN):
            val -= intactdoublecorner

        bm = bk = wm = wk = 0
        for pos in CENTER:
            if board[pos] == BLACK|MAN: bm += 1
            elif board[pos] == BLACK|KING: bk += 1
            elif board[pos] == WHITE|MAN: wm += 1
            elif board[pos] == WHITE|KING: wk += 1

        val += (bm - wm) * mcv
        val += (bk - wk) * kcv

        bm = bk = wm = wk = 0
        for pos in EDGE:
            if board[pos] == BLACK|MAN: bm += 1
            elif board[pos] == BLACK|KING: bk += 1
            elif board[pos] == WHITE|MAN: wm += 1
            elif board[pos] == WHITE|KING: wk += 1

        val += (bm - wm) * mev
        val += (bk - wk) * kev

        tempo = 0
        for i in xrange(8):
            for j in xrange(8):
                if board[i,j] == BLACK|MAN:
                    tempo += i
                elif board[i,j] == WHITE|MAN:
                    tempo -= 7-i
        if nm >= 16:
            val += opening * tempo
        if 12 <= nm <= 15:
            val += midgame * tempo
        if nm < 9:
            val += endgame * tempo

        for pos in SAFEEDGE:
            if nbk + nbm > nwk + nwm and nwk < 3:
                if board[pos] == (WHITE|KING):
                    val -= 15
            
            if nwk + nwm > nbk + nbm and nbk < 3:
                if board[pos] == (BLACK|KING):
                    val += 15

        # I have no idea what this last bit does.. :(
        stonesinsystem = 0
        if nwm + nwk - nbk - nbm == 0:
            if game.to_move == BLACK:
                for row in xrange(0, 8, 2):
                    for c in xrange(0, 8, 2):
                        if board[row,c] != FREE:
                            stonesinsystem += 1
                if stonesinsystem % 2:
                    if nm + nk <= 12: val += 1
                    if nm + nk <= 10: val += 1
                    if nm + nk <= 8: val += 2
                    if nm + nk <= 6: val += 2
                else:
                    if nm + nk <= 12: val -= 1
                    if nm + nk <= 10: val -= 1
                    if nm + nk <= 8: val -= 2
                    if nm + nk <= 6: val -= 2
            else:
                for row in xrange(1, 8, 2):
                    for c in xrange(1, 8, 2):
                        if board[row,c] != FREE:
                            stonesinsystem += 1
                if stonesinsystem % 2:
                    if nm + nk <= 12: val += 1
                    if nm + nk <= 10: val += 1
                    if nm + nk <= 8: val += 2
                    if nm + nk <= 6: val += 2
                else:
                    if nm + nk <= 12: val -= 1
                    if nm + nk <= 10: val -= 1
                    if nm + nk <= 8: val -= 2
                    if nm + nk <= 6: val -= 2

        return val

class PieceCounter(object):
    def evaluate(self, game):
        counts = np.bincount(game.board.flat)

        nwm = counts[WHITE|MAN]
        nwk = counts[WHITE|KING]
        nbm = counts[BLACK|MAN]
        nbk = counts[BLACK|KING]

        vb = (100 * nbm + 130 * nbk)
        vw = (100 * nwm + 130 * nwk)

        return vb - vw

class NetworkHeuristic(object):
    """ Heuristic based on feeding the board state to a neural network
    """
    def __init__(self, network):
        self.network = network

    def evaluate(self, game):
        if game.game_over():
            return 5000 * game.winner()

        net_inputs = ((game.board == BLACK | MAN) * 0.5 +
                      (game.board == WHITE | MAN) * -0.5 +
                      (game.board == BLACK | KING) * 0.75 +
                      (game.board == WHITE | KING) * -0.75)
        self.network.flush()
        # Feed twice to propagate through 3 layer network:
        value = self.network.feed(net_inputs, add_bias=False)
        return value[-1]

class RandomOpponent(object):
    """ An opponent that plays random moves """
    def pickmove(self, game):
        return random.choice(list(game.all_moves()))

class Checkers(object):
    """ Represents the checkers game(state)
    """

    def __init__(self, no_advance_draw=50, fly_kings=False, minefield=False, max_repeat_moves=3):
        """ Initialize the game board. """
        self.no_advance_draw = no_advance_draw

        self.board = NUMBERING.copy() #: The board state
        self.to_move = BLACK          #: Whose move it is
        self.turn = 0
        self.history = []
        self.advancement = []
        self.minefield = minefield
        self.fly_kings = fly_kings
        self.max_repeat_moves = max_repeat_moves

        tiles = self.board > 0
        self.board[tiles] = EMPTY
        self.board[:3,:] = BLACK | MAN
        # self.board[3, :] = WHITE | MAN
        self.board[5:,:] = WHITE | MAN
        self.board[np.logical_not(tiles)] = FREE
        
        self._moves = None

    def all_moves(self):
        if self._moves is None:
            self._moves = list(self.generate_moves())
        # Remove repeated moves
        if (len(self.history) > self.max_repeat_moves * 4 and 
            len(self._moves) > 1 and
            self.history[-4] in self._moves):
            for i in range(self.max_repeat_moves):
                if (self.history[-2 - 4 * i] != self.history[-2 - 4 * (i+1)]):
                    break
            if i == self.max_repeat_moves - 1:
                self._moves.remove(self.history[-4])
                # print "REMOVED" + str(self.history[-4])
        return self._moves
                       
    def generate_moves(self):
        """ Return a list of possible moves. """
        captures_possible = False
        pieces = []
        # Check for possible captures first:
        for n, (y,x) in INVNUM.iteritems():
            piece = self.board[y, x]
            if piece & self.to_move:
                pieces.append((n, (y, x), piece))
                for m in self.captures((y, x), piece, self.board):
                    if len(m) > 1:
                        captures_possible = True
                        yield m
        # Otherwise check for normal moves:
        if not captures_possible:
            for (n, (y, x), piece) in pieces:
                # MAN moves
                if piece & MAN:
                    nextrow = y + 1 if (self.to_move == BLACK) else y - 1
                    if 0 <= nextrow < 8:
                        if x - 1 >= 0 and self.board[nextrow, x - 1] == EMPTY:
                            yield (n, NUMBERING[nextrow, x - 1])
                        if x + 1 < 8 and self.board[nextrow, x + 1] == EMPTY:
                            yield (n, NUMBERING[nextrow, x + 1])
                # KING moves
                else:
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:                
                            dist = 1
                            while True:
                                tx, ty = x + dist * dx, y + dist * dy # Target square
                                if not ((0 <= tx < 8 and 0 <= ty < 8) and self.board[ty, tx] == EMPTY):
                                    break
                                else:
                                    yield (n, NUMBERING[ty, tx])
                                if not self.fly_kings:
                                    break
                                dist += 1

    
    def captures(self, (py, px), piece, board, captured=[], start=None):
        """ Return a list of possible capture moves for given piece in a 
            checkers game. 

            :param (py, px): location of piece on the board
            :param piece: piece type (BLACK/WHITE|MAN/KING)
            :param board: the 2D board matrix
            :param captured: list of already-captured pieces (can't jump twice)
            :param start: from where this capture chain started.
            """
        if start is None:
            start = (py, px)
        opponent = BLACK if piece & WHITE else WHITE
        forward = [-1, 1] if piece & KING else [1] if piece & BLACK else [-1]
        # Look for capture moves
        for dx in [-1, 1]:
            for dy in forward:
                jx, jy = px, py
                while True:
                    jx += dx # Jumped square
                    jy += dy 
                    # Check if piece at jx, jy:
                    if not (0 <= jx < 8 and 0 <= jy < 8):
                        break
                    if board[jy, jx] != EMPTY:
                        tx = jx + dx # Target square
                        ty = jy + dy 
                        # Check if it can be captured:
                        if ((0 <= tx < 8 and 0 <= ty < 8) and
                            ((ty, tx) == start or board[ty, tx] == EMPTY) and
                            (jy, jx) not in captured and
                            (board[jy, jx] & opponent)
                            ):
                            # Normal pieces cannot continue capturing after reaching last row
                            if not piece & KING and (piece & WHITE and ty == 0 or piece & BLACK and ty == 7):
                                yield (NUMBERING[py, px], NUMBERING[ty, tx])
                            else:
                                for sequence in self.captures((ty, tx), piece, board, captured + [(jy, jx)], start):
                                    yield (NUMBERING[py, px],) + sequence
                        break
                    else:
                        if piece & MAN or not self.fly_kings:
                            break
        yield (NUMBERING[py, px],)
                        
        
    def play(self, move):
        """ Play the given move on the board. """
        if move not in self.all_moves():
            raise IllegalMoveError("Illegal move")
        self.history.append(move)
        positions = [INVNUM[p] for p in move]
        (ly, lx) = positions[0]
        # Check for captures
        capture = False
        stone_dies = False
        for (py, px) in positions[1:]:
            ydir = 1 if py > ly else -1
            xdir = 1 if px > lx else -1
            for y, x in zip(xrange(ly + ydir, py, ydir),xrange(lx + xdir, px, xdir)):
                if self.board[y,x] != EMPTY:
                    self.board[y,x] = EMPTY
                    if self.minefield and 2 <= x < 6 and 2 <= y < 6:
                        stone_dies = True
                    capture = True
            (ly, lx) = (py, px)
        # Move the piece
        (ly, lx) = positions[0]
        (py, px) = positions[-1]
        piece = self.board[ly, lx]
        self.board[ly, lx] = EMPTY
        # The game advances if a pieces is captured, or if a man moves
        # towards the kings row.
        self.advancement.append(capture or (piece & MAN))
        # Check if the piece needs to be crowned
        if (piece & MAN) and ((piece & BLACK and py == 7) or (piece & WHITE and py == 0)):
            piece = piece ^ MAN | KING
        self.board[py, px] = piece

        # Kill the piece if a capture was performed on the minefield.
        if stone_dies:
            self.board[py, px] = EMPTY

        self.to_move = WHITE if self.to_move == BLACK else BLACK
        self.turn += 1
        # Cached moveset is invalidated.
        self._moves = None
        return self

    def copy_and_play(self, move):
        return self.copy().play(move)

    def check_draw(self, verbose=False):
        # If there were no captures in the last [50] moves, draw.
        i = 0
        for i in xrange(len(self.advancement)):
            if self.advancement[-(i+1)]:
                break
        if verbose:
            print "Last capture: %d turns ago." % (i)
        return (i > self.no_advance_draw)
        
    def game_over(self):
        """ Whether the game is over. """
        if self.check_draw():
            return True
        for move in self.all_moves():
            # If the iterator returns any moves at all, the game is not over.
            return False
        # Otherwise it is.
        return True
        
    def winner(self):
        """ Returns board score. """
        if self.check_draw():
            if (self.board & MAN).sum() == 0:
                w = (self.board & WHITE).sum()
                b = (self.board & BLACK).sum()
                if w >= 3 * b:
                    return -1.0
                elif b >= 3* w:
                    return 1.0
            return 0.0
        if not self.game_over():
            return 0.0
        else:
            return 1.0 if self.to_move == WHITE else -1.0

    def copy(self):
        new = copy.copy(self)               # Copy all.
        new.board = self.board.copy()       # Copy the board explicitly
        new._moves = copy.copy(self._moves) # Shallow copy is enough.
        new.history = self.history[:]
        new.advancement = self.advancement[:]
        return new

    def __str__(self):
        s = np.array([l for l in "-    wb  WB      "])
        s = s[self.board]
        if self.to_move == BLACK:
            s[0,7] = 'v'
        else:
            s[7,0] = '^'
        n = [''.join(('%2d' % n) if (n > 0) else '  ' for n in row) for row in NUMBERING]
        s = [' '.join(l) for l in s]
        o = ['   %s   ||   %s' % line for line in zip(s, n)]
        o = '\n'.join(o[::-1])
        return o
    
### PROCEDURE ###

if __name__ == '__main__':
    opponent = HeuristicOpponent(SimpleHeuristic(), search_depth=4)
    opponent.play_against(user_side=BLACK)
    