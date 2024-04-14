# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
"""

import numpy as np
import copy


def distance_policy_value_fn(board):
    """
    게임 보드와 마지막 무브 인덱스를 받아, 해당 무브 주변의 수에 높은 확률을,
    먼 수에는 낮은 확률을 부여하는 Policy Value Function
    """
    action_probs = np.zeros(len(board.availables))
    last_move = board.last_move
    if last_move == -1:  # 첫 수인 경우
        action_probs = np.ones(len(board.availables)) / len(board.availables)
        return (zip(board.availables, action_probs)), 0

    last_move_location = board.move_to_location(last_move)

    # 각 가능한 수에 대해 거리에 따라 가중치 계산
    for i, move in enumerate(board.availables):
        move_location = board.move_to_location(move)
        # checkerboard 거리 계산
        distance = max(abs(move_location[0] - last_move_location[0]), abs(move_location[1] - last_move_location[1]))
        # 거리가 가까울수록 높은 확률 부여
        weight = np.exp(-distance) 
        action_probs[i] = weight

    # 확률 값들을 정규화
    total = np.sum(action_probs)
    if total > 0:
        action_probs /= total
    else:
        action_probs = np.ones(len(board.availables)) / len(board.availables)  # 만약 계산 오류가 발생하면 균등 확률 부여
    return list(zip(board.availables, action_probs)), 0  # 점수는 0으로 고정, 실제 게임 상황에 따라 조절 가능


def distance_policy_value_fn_1(board):
    """
    게임 보드와 마지막 무브 인덱스를 받아, 해당 무브 주변의 수에 높은 확률을,
    먼 수에는 낮은 확률을 부여하는 Policy Value Function
    """
    action_probs = np.zeros(len(board.availables))
    last_move = board.last_move
    if last_move == -1:  # 첫 수인 경우
        action_probs = np.ones(len(board.availables)) / len(board.availables)
        return (zip(board.availables, action_probs)), 0

    last_move_location = board.move_to_location(last_move)

    # 각 가능한 수에 대해 거리에 따라 가중치 계산
    for i, move in enumerate(board.availables):
        move_location = board.move_to_location(move)
        # distance 계산
        x_dis = abs(move_location[0] - last_move_location[0])
        y_dis = abs(move_location[1] - last_move_location[1])
        # 마지막 수로부터 가로, 세로, 대각선 방향만 search
        if (x_dis == y_dis or x_dis == 0 or y_dis == 0):
            distance = max(x_dis, y_dis)
            if distance <= 4:
                action_probs[i] = 1
                if distance == 1:
                    action_probs[i] = 2
            else: 
                action_probs[i] = 0
        else:
            action_probs[i] = 0
    # 확률 값들을 정규화
    total = np.sum(action_probs)
    if total > 0:
        action_probs /= total
    else:
        action_probs = np.ones(len(board.availables)) / len(board.availables)  # 만약 계산 오류가 발생하면 균등 확률 부여
    return list(zip(board.availables, action_probs)), 0  # 점수는 0으로 고정, 실제 게임 상황에 따라 조절 가능


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # Evaluate the leaf node by rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs, _ = distance_policy_value_fn(state)
            actions, probabilities = zip(*action_probs)  # 동작과 확률을 분리

            # numpy의 random.choice를 사용하여 주어진 확률 분포에 따라 동작을 랜덤하게 선택
            selected_action = np.random.choice(actions, p=probabilities)

            # 선택된 동작을 사용
            state.do_move(selected_action)

            # max_action = max(action_probs, key=itemgetter(1))[0]
            # state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(distance_policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        # AI가 첫 수를 둘 경우, 중앙으로 첫 수 고정
        if len(sensible_moves) == board.width * board.height:
            move = board.width * board.height // 2
            self.mcts.update_with_move(-1)
            return move
        
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
