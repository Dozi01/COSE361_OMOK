# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
from game import Board, Game
from mcts import MCTSPlayer


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 19, 19
    time_limit = 0

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        mcts_player = MCTSPlayer(c_puct=3, n_playout=100)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        start_player = int(input("Select the first player\n Human = 0, AI = 1 : "))
        time_limit = int(input("Insert the time limit for each move, 0 for infinite \n time : "))
        game.start_play(human, mcts_player, start_player=start_player, is_shown=1, time_limit = time_limit)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
