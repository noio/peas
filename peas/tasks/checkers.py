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
BLACK_MAN = BLACK|MAN
WHITE_MAN = WHITE|MAN
BLACK_KING = BLACK|KING
WHITE_KING = WHITE|KING

NUMBERING = np.array([[ 4,  0,  3,  0,  2,  0,  1,  0],
                      [ 0,  8,  0,  7,  0,  6,  0,  5],
                      [12,  0, 11,  0, 10,  0,  9,  0],
                      [ 0, 16,  0, 15,  0, 14,  0, 13],
                      [20,  0, 19,  0, 18,  0, 17,  0],
                      [ 0, 24,  0, 23,  0, 22,  0, 21],
                      [28,  0, 27,  0, 26,  0, 25,  0],
                      [ 0, 32,  0, 31,  0, 30,  0, 29]])

# The internal board representation is like this, because it makes simulating
# moves easy, every move is either +4, +5, -4 or -5. And moving "off the board",
# still results in a valid array index, albeit one marked as "EMPTY" (in stead
# of "FREE"). This is why the internal board is size 46, to allow 40+5.

INTERNAL = np.array([[ 5,  0,  6,  0,  7,  0,  8,  0],
                     [ 0,  10, 0, 11,  0, 12,  0, 13],
                     [14,  0, 15,  0, 16,  0, 17,  0],
                     [ 0, 19,  0, 20,  0, 21,  0, 22],
                     [23,  0, 24,  0, 25,  0, 26,  0],
                     [ 0, 28,  0, 29,  0, 30,  0, 31],
                     [32,  0, 33,  0, 34,  0, 35,  0],
                     [ 0, 37,  0, 38,  0, 39,  0, 40]])

ALL_SQUARES = list(np.unique(INTERNAL[INTERNAL>0]))

# CENTER = [(2,2), (2,4), (3,3), (3,5), (4,2), (4,4), (5,3), (5,5)]
CENTER = [15, 16, 20, 21, 24, 25, 29, 30]
# EDGE = [(0,0), (0,2), (0,4), (0,6), (1,7), (2,0), (3,7), (4,0), (5,7), (6,0), (7,1), (7,3), (7,5), (7,7)]
EDGE = [5, 6, 7, 8, 13, 14, 22, 23, 31, 32, 37, 38, 39, 40]
# SAFEEDGE = [(0,6), (1,7), (6,0), (7,1)]
SAFEEDGE = [7, 13, 32, 37]
ROW = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,0,3,3,3,3,4,4,4,4,0,5,5,5,5,6,6,6,6,0,7,7,7,7]

INV_NUMBERING =[tuple(a[0] for a in np.nonzero(NUMBERING == n)) for n in range(1, NUMBERING.max() + 1)]
INV_INTERNAL = [(tuple(a[0] for a in np.nonzero(INTERNAL == n)) if n in INTERNAL else None) for n in range(INTERNAL.max() + 1)]
INTERNAL_TO_NUMBERING = [NUMBERING[a] for a in INV_INTERNAL]
NUMBERING_TO_INTERNAL = [INTERNAL[a] for a in INV_NUMBERING]


### EXCEPTIONS

class IllegalMoveError(Exception):
    pass

### FUNCTIONS ###

def board2d(internal):
    """ Convert internal (46) board representation to 8x8 
    """
    board = np.zeros((8,8), dtype=int)
    for i in ALL_SQUARES:
        board[INV_INTERNAL[i]] = internal[i]
    return board

def movestr(move):
    """ Print a move as human readable (using NUMBERING)
    """
    if move is None: return None
    # Check if it's a capture:
    n = [str(INTERNAL_TO_NUMBERING[m]) for m in move]
    if abs(move[1]-move[0]) > 5 or len(move) > 2:
        return 'x'.join(n)
    else:
        return '-'.join(n)

def alphabeta(node, heuristic, player_max=True, depth=4, alpha=-inf, beta=inf, killer_moves=defaultdict(set), num_evals=[0]):
    """ Performs alphabeta search.
        From wikipedia pseudocode.
    """
    if node.game_over():
        return heuristic(node)

    moves = node.all_moves()
    
    if depth == 0:
        # Extend the search for capture moves (this is what simplech does)
        if node.captures_possible:
            depth = 1
        else:
            num_evals[0] += 1
            value = heuristic(node, False)
            # if value == 25: heuristic(node, True)
            # print "[%s] %.3f" % ("+" if player_max else "-", value),
            return value
    pmx = not player_max

    killers = killer_moves[node.turn+1]
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

def playgame(black, white, game=None, history=None, verbose=False):
    """ Play a game between two opponents, return stats """
    # Input arguments
    if game is None:
        game = Checkers()
    # Initialize
    fitness = []
    i = 0
    current, next = black, white

    print "Checkers: %s vs. %s " % (black, white)
    while not game.game_over():
        i += 1
        historical = history.pop(0) if history else None
        move = current.pickmove(game, historical=historical, verbose=verbose)
        if move != historical:
            history = None
        game.play(move)
        fitness.append(gamefitness(game))
        sys.stdout.write('.')
        sys.stdout.flush()
        current, next = next, current

    print
    print game
    # Fitness over last 100 episodes
    fitness.extend([gamefitness(game)] * (100 - len(fitness)))
    fitness = fitness[-100:]
    winner = game.winner()
    return winner, fitness

### CLASSES ###

class CheckersTask(object):
    """ Represents a checkers game played by an evolved phenotype against
        a fixed opponent.
    """
    def __init__(self, search_depth=4, 
                       opponent='simplech', 
                       opponent_search_depth=4, 
                       opponent_handicap=0.0, 
                       win_to_solve=3):
        self.search_depth = search_depth
        self.opponent_search_depth = opponent_search_depth
        self.opponent_handicap = opponent_handicap
        self.opponent = opponent
        self.win_to_solve = win_to_solve

    def evaluate(self, network):
        # Setup
        game = Checkers()
        player = HeuristicOpponent(NetworkHeuristic(network), search_depth=self.search_depth)
        if self.opponent == 'simplech':
            opponent = HeuristicOpponent(SimpleHeuristic(), search_depth=self.opponent_search_depth, handicap=self.opponent_handicap)
        elif self.opponent == 'counter':
            opponent = HeuristicOpponent(PieceCounter(), search_depth=self.opponent_search_depth, handicap=self.opponent_handicap)
        elif self.opponent == 'random':
            opponent = RandomOpponent()
        # Play the game
        winner, fitness = playgame(player, opponent, game)
        won = winner >= 1.0
        draw = winner == 0.0
        score = sum(fitness)
        if won:
            score += 30000
        turns = len(game.history)
        print "\nGame finished in %d turns. Winner: %s. Score: %d. Final state fitness: %d." % (turns,game.winner(), score, fitness[-1])
        return {'fitness':score, 'won': won, 'draw': draw, 'turns': turns, '_move_history': game.history[:]}

    def play_against(self, network, user_side=WHITE, history=None):
        # Setup
        game = Checkers()
        player = HeuristicOpponent(NetworkHeuristic(network), search_depth=self.search_depth)
        user = UserOpponent(auto = HeuristicOpponent(SimpleHeuristic(), search_depth=self.opponent_search_depth))
        playgame(player, user, game=game, history=history, verbose=True)

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
                value = network.feed(inpt, add_bias=False)[-1]
                output[y,x] = value
        plt.imshow(output, vmin=-1, vmax=1, interpolation='nearest', extent=[0,8,0,8], cmap='RdYlGn')
        plt.grid(zorder=2)
        plt.savefig(filename)
        print filename
        plt.close()
            
class UserOpponent(object):
    
    def __init__(self, auto=None):
        self.auto = auto
        if self.auto is None:
            self.auto = HeuristicOpponent(SimpleHeuristic(), search_depth=4)
        self.skip = False

    def pickmove(self, board, historical=None, verbose=None):
        best = self.auto.pickmove(board, verbose=True)
        moved = False
        while not moved:
            # Prompt user for move
            print board
            if board.history:
                print "Opponent moved %s" % movestr(board.history[-1])
            print "Enter move ([h]istory: %s, [a]uto: %s):" % (movestr(historical), movestr(best)),
            try:
                if not self.skip:
                    user_input = raw_input()
                else:
                    user_input = 'a'
                if user_input == 's':
                    self.skip = True
                if user_input == 'h' or (historical is not None and user_input == ''):
                    move = historical
                elif self.skip or user_input == 'a' or (not historical and user_input == ''):
                    move = best
                elif user_input == 'q':
                    return None
                elif ' ' in user_input:
                    move = tuple(int(NUMBERING_TO_INTERNAL[i]) for i in user_input.split(' '))
                else:
                    move = tuple(int(NUMBERING_TO_INTERNAL[i]) for i in user_input.split('-'))
                if move is not None:
                    board.copy_and_play(move)
                moved = True
            except (IllegalMoveError, ValueError):
                print "Illegal move."
        return move

        
class HeuristicOpponent(object):
    """ Opponent that utilizes a heuristic combined with alphabeta search
        to decide on a move.
    """
    def __init__(self, heuristic, search_depth=4, handicap=0.0):
        self.search_depth = search_depth
        self.heuristic = heuristic
        self.handicap = handicap
        self.killer_moves = defaultdict(set)
    
    def pickmove(self, board, verbose=False, historical=None):
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
        for move in moves:
            val = alphabeta(board.copy_and_play(move), self.heuristic.evaluate, 
                depth=self.search_depth-1, player_max=not player_max, killer_moves=self.killer_moves, num_evals=evals)
            if (player_max and val > bestval) or (not player_max and val < bestval):
                bestval = val
                secondbest = bestmove
                bestmove = move
        # Pick second best move
        if verbose: 
            # self.heuristic.evaluate(board.copy_and_play(bestmove), verbose=True)
            print "%d evals. value: %.2f" % (evals[0], bestval)
        if historical and self.handicap == 0 and bestmove != historical:
            raise Exception("Playing different move (%s) from history (%s). Shouldn't happen because I'm deterministic!" %
                (bestmove, historical))
        if secondbest is not None and self.handicap > 0 and random.random() < self.handicap:
            return secondbest
        return bestmove

    def __str__(self):
        return '%s with %s (lookahead: %d, handicap: %.2f)' % (self.__class__.__name__,
            self.heuristic.__class__.__name__, self.search_depth, self.handicap)
        
class SimpleHeuristic(object):
    """ Simple piece/position counting heuristic, adapted from simplech
    """
    def evaluate(self, game, verbose=False):
        if verbose:
            print "EVALUTING: "
            print game


        if game.game_over():
            return 5000 * game.winner()
        board = game.board
        counts = np.bincount(board)

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

        nwm = counts[WHITE_MAN]
        nwk = counts[WHITE_KING]
        nbm = counts[BLACK_MAN]
        nbk = counts[BLACK_KING]

        if verbose: print nwm, nwk, nbm, nbk

        vb = (100 * nbm + 130 * nbk)
        vw = (100 * nwm + 130 * nwk)
        
        val = 0
        
        if (vb + vw) > 0:        
            val = (vb - vw) + (250 * (vb-vw))/(vb+vw); #favor exchanges if in material plus

        if verbose: print 'counts', val

        nm = nwm + nbm
        nk = nwk + nbk

        val += turn if game.to_move == BLACK else -turn

        if board[23] == (BLACK_MAN) and board[28] == (WHITE_MAN):
            val += cramp
        if board[22] == (WHITE_MAN) and board[17] == (BLACK_MAN):
            val -= cramp

        # Back rank guard
        code = 0
        if (board[5] & MAN): code += 1
        if (board[6] & MAN): code += 2
        if (board[7] & MAN): code += 4
        if (board[8] & MAN): code += 8
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
        if (board[37] & MAN): code += 8
        if (board[38] & MAN): code += 4
        if (board[39] & MAN): code += 2
        if (board[40] & MAN): code += 1

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

        if verbose: print 'backrank', val

        # Double Corner
        if board[8] == BLACK_MAN and (board[12] == BLACK_MAN or board[13] == BLACK_MAN):
            val += intactdoublecorner
        if board[37] == WHITE_MAN and (board[32] == WHITE_MAN or board[33] == WHITE_MAN):
            val -= intactdoublecorner

        if verbose: print 'double corner', val

        # Center control
        bm = bk = wm = wk = 0
        for pos in CENTER:
            if board[pos] == BLACK_MAN: bm += 1
            elif board[pos] == BLACK_KING: bk += 1
            elif board[pos] == WHITE_MAN: wm += 1
            elif board[pos] == WHITE_KING: wk += 1

        val += (bm - wm) * mcv
        val += (bk - wk) * kcv

        if verbose: print 'center control', val

        # Edge
        bm = bk = wm = wk = 0
        for pos in EDGE:
            if board[pos] == BLACK_MAN: bm += 1
            elif board[pos] == BLACK_KING: bk += 1
            elif board[pos] == WHITE_MAN: wm += 1
            elif board[pos] == WHITE_KING: wk += 1

        val -= (bm - wm) * mev
        val -= (bk - wk) * kev

        if verbose: print 'edge', val

        # Tempo
        tempo = 0
        for i in range(5, 41):
            if board[i] == BLACK_MAN:
                tempo += ROW[i]
            elif board[i] == WHITE_MAN:
                tempo -= 7-ROW[i]
        if nm >= 16:
            val += opening * tempo
        if 12 <= nm <= 15:
            val += midgame * tempo
        if nm < 9:
            val += endgame * tempo

        for pos in SAFEEDGE:
            if nbk + nbm > nwk + nwm and nwk < 3:
                if board[pos] == (WHITE_KING):
                    val -= 15
            
            if nwk + nwm > nbk + nbm and nbk < 3:
                if board[pos] == (BLACK_KING):
                    val += 15

        if verbose: print 'tempo', val

        # [[ 5  0  6  0  7  0  8  0]
        #  [ 0 10  0 11  0 12  0 13]
        #  [14  0 15  0 16  0 17  0]
        #  [ 0 19  0 20  0 21  0 22]
        #  [23  0 24  0 25  0 26  0]
        #  [ 0 28  0 29  0 30  0 31]
        #  [32  0 33  0 34  0 35  0]
        #  [ 0 37  0 38  0 39  0 40]]

        # I have no idea what this last bit does XD.. it's correct tho.
        stonesinsystem = 0
        ns = nm + nk
        if nwm + nwk - nbk - nbm == 0:
            if game.to_move == BLACK:
                for i in xrange(5, 9):
                    for j in xrange(4):
                        if board[i + 9 * j] != FREE:
                            stonesinsystem += 1
                if stonesinsystem % 2 == 0:
                    if ns <= 12: val += 1
                    if ns <= 10: val += 1
                    if ns <= 8: val += 2
                    if ns <= 6: val += 2
                else:
                    if ns <= 12: val -= 1
                    if ns <= 10: val -= 1
                    if ns <= 8: val -= 2
                    if ns <= 6: val -= 2
            else:
                for i in xrange(10, 14):
                    for j in xrange(4):
                        if board[i + 9 * j] != FREE:
                            stonesinsystem += 1
                if stonesinsystem % 2 == 0:
                    if ns <= 12: val += 1
                    if ns <= 10: val += 1
                    if ns <= 8: val += 2
                    if ns <= 6: val += 2
                else:
                    if ns <= 12: val -= 1
                    if ns <= 10: val -= 1
                    if ns <= 8: val -= 2
                    if ns <= 6: val -= 2

        if verbose: print "Value: %.3f" % val
        return val

class PieceCounter(object):
    def evaluate(self, game, verbose=False):
        if game.game_over():
            return 5000 * game.winner()

        counts = np.bincount(game.board)

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

    def evaluate(self, game, verbose=False):
        if game.game_over():
            return 5000 * game.winner()
        board = board2d(game.board)
        net_input = np.zeros((8,8))
        net_input[board == BLACK | MAN] = 0.5
        net_input[board == WHITE | MAN] = -0.5
        net_input[board == BLACK | KING] = 0.75
        net_input[board == WHITE | KING] = -0.75

        # Feed twice to propagate through 3 layer network:
        value = self.network.feed(net_input, add_bias=False)
        return value[-1]

class RandomOpponent(object):
    """ An opponent that plays random moves """
    def pickmove(self, game, historical=None):
        return random.choice(list(game.all_moves()))

class Checkers(object):
    """ Represents the checkers game(state)
    """

    def __init__(self, no_advance_draw=50, max_repeat_moves=1000):
        """ Initialize the game board. """
        self.no_advance_draw = no_advance_draw

        self.to_move = BLACK          #: Whose move it is
        self.turn = 0
        self.history = []
        self.history_captures = []
        self.advancement = []
        self.max_repeat_moves = max_repeat_moves

        self.board = np.zeros(46, dtype=int)
        self.board[5:41] = FREE
        self.board[INTERNAL[:3,:]] = BLACK | MAN
        # self.board[3, :] = WHITE | MAN
        self.board[INTERNAL[5:,:]] = WHITE | MAN
        self.board[[0,9,18,27,36]] = EMPTY
        
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
        self.captures_possible = False
        pieces = []
        # Check for possible captures first:
        for s in ALL_SQUARES:
            piece = self.board[s]
            if piece & self.to_move:
                pieces.append((s, piece))
                for m in self.captures(s, piece, self.board):
                    if len(m) > 1:
                        self.captures_possible = True
                        yield m
        # Otherwise check for normal moves:
        if not self.captures_possible:
            for (s, piece) in pieces:
                # MAN moves
                if piece & MAN:
                    n1, n2 = (s + 4, s + 5) if self.to_move == BLACK else (s - 4, s - 5)
                    if self.board[n1] == FREE: yield (s, n1)
                    if self.board[n2] == FREE: yield (s, n2)
                # KING moves
                else:
                    for m in (s + 4, s + 5, s - 4, s - 5):
                        if self.board[m] == FREE:
                            yield (s, m)
    
    def captures(self, s, piece, board, captured=[], start=None):
        """ Return a list of possible capture moves for given piece in a 
            checkers game. 

            :param s: location of piece on the board
            :param piece: piece type (BLACK/WHITE|MAN/KING)
            :param board: the 2D board matrix
            :param captured: list of already-captured pieces (can't jump twice)
            :param start: from where this capture chain started.
            """
        if start is None:
            start = s
        opponent = BLACK if piece & WHITE else WHITE
        dirs = [-5, -4, 4, 5] if piece & KING else [4, 5] if piece & BLACK else [-5, -4]
        # Look for capture moves
        for d in dirs:
            j = s + d
            # Check if piece at [j]umped square:
            if board[j] & opponent:
                t = j + d # [t]arget square (where we land after jump)
                # Check if it can be captured:
                if (board[t] == FREE and j not in captured):
                    # Normal pieces cannot continue capturing after reaching last row (5,6,7,8 or 37,38,39,40)
                    if not piece & KING and (piece & WHITE and t <= 8 or piece & BLACK and t >= 37):
                        yield (s, t)
                    else:
                        for sequence in self.captures(t, piece, board, captured + [j], start):
                            yield (s,) + sequence
        yield (s,)
                        
        
    def play(self, move):
        """ Play the given move on the board. """
        if move not in self.all_moves():
            raise IllegalMoveError("Illegal move")
        self.history.append(move)
        # Check for captures
        captured = []
        l = move[0]
        for p in move[1:]:
            # If move is a capture
            if abs(p - l) > 5:
                c = (p + l) / 2
                captured.append(self.board[c])
                self.board[c] = FREE
            l = p
        self.history_captures.append(captured)
        # Move the piece
        piece = self.board[move[0]]
        self.board[move[0]] = FREE
        # The game advances if a pieces is captured, or if a man moves
        # towards the kings row.
        self.advancement.append(captured or (piece & MAN))
        # Check if the piece needs to be crowned
        if (piece & MAN) and ((piece & BLACK and p >= 37) or (piece & WHITE and p <= 8)):
            piece = piece ^ MAN | KING
        self.board[p] = piece
        self.to_move = WHITE if self.to_move == BLACK else BLACK
        self.turn += 1
        # Cached moveset is invalidated.
        self._moves = None
        return self

    def undoplay(self, move):
        raise NotImplemented
        if move != self.history[-1]:
            raise Exception("Trying to undo move that wasn't last.")
        # Restore captured pieces
        for i, capped in enumerate(self.history_captures.pop()):
            # Captured piece location is average of move locations
            self.board[move[i]+move[i+1] / 2] = capped
        # Undo the actual move:
        piece = self.board[move[-1]]
        self.board[move[-1]] = FREE
        self.board[move[0]] = piece
        self.history.pop()
        self.to_move = WHITE if self.to_move == BLACK else BLACK
        self.turn -= 1
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
        arr = np.array(self.board)
        if self.check_draw():
            wk = (arr & (WHITE|KING)).sum()
            bk = (arr & (BLACK|KING)).sum()
            if wk >= 3 * bk:
                return -1.0
            elif bk >= 3* wk:
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
        new.history_captures = self.history_captures[:]
        new.advancement = self.advancement[:]
        return new

    def __str__(self):
        board = board2d(self.board)
        s = np.array([l for l in "     wb  WB     -"])
        s = s[board]
        if self.to_move == BLACK:
            s[0,7] = '^'
        else:
            s[7,0] = 'v'
        n = [''.join(('%2d' % n) if (n > 0) else '  ' for n in row) for row in NUMBERING]
        s = [' '.join(l) for l in s]
        o = ['   %s   ||   %s' % line for line in zip(s, n)]
        o = '\n'.join(o[::-1])
        return o
    
### PROCEDURE ###

if __name__ == '__main__':
    c = Checkers()
    user = UserOpponent()
    opponent = HeuristicOpponent(SimpleHeuristic(), search_depth=4)
    playgame(opponent, user)
    # opponent.play_against(user_side=BLACK)
    