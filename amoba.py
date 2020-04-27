
import pygame, sys
from pygame.locals import *
from pygame import gfxdraw  # this has anti-aliasing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# we are representing the board with it resolution, how much square we divide it into (square/column), and from that
# defining a board_rep which will contain the coordinates of the left upper edge of the squares

class BoardRepresentation:
    """Representing the board as a physical object, which we can draw on."""

    def __init__(self, resolution, square_num):
        self.resolution = resolution  # resolution in pixels
        self.square_num_row = square_num  # number of rows
        self.square_num_col = int(resolution[1] / resolution[
            0] * square_num)  # number of clumns, computed so the board will be devided into squares
        self.board_rep = np.array(
            self.board_making())  # board_rep is a 2d list of the coordinates of each square's upper left edge
        self.length = self.board_rep[0, 1][1] - self.board_rep[0, 0][1]  # length of the side of a square
        self.winning_rectangle = tuple()  # this will contain the position of the square, where the winning
        #   move was made, if the winning move was made in the square in the 4. row, and 2. column:
        #   [3,1] would be the value passed to it

    def board_making(self):  # setting the board_rep
        temp = list()
        for i in range(self.square_num_row):
            temp_inner = list()
            for ii in range(self.square_num_col):
                temp_inner.append([int(i * self.resolution[0] / self.square_num_row),
                                   int(ii * self.resolution[1] / self.square_num_col)])
            temp.append(temp_inner)
        return tuple(temp)  # we want the constants immutable

    # next function gets x, and y coordinates of the surface we are playing on, in this case later we will use the
    # coordinates of the cursor, the function returns which square the coordinates represent
    # lets say the cursor is over the square in the 3. row and 4. column, we want to return (2,3) then
    def which_element(self, x_coord, y_coord):
        for i, rows in enumerate(self.board_rep):
            for ii, cols in enumerate(rows):
                if cols[0] <= x_coord <= cols[0] + self.length:
                    if cols[1] <= y_coord <= cols[1] + self.length:
                        return i, ii
        pass


# inherintence may not needed here, but whatever
class AbstractRepresentation(BoardRepresentation):
    """ Abstract representation of the board, as a matrix of 1s, and -1s where a player made a move,
     and 0s for empty squares"""

    def __init__(self, resolution, square_num, numofwin):
        super().__init__(resolution, square_num)
        self.board_matrix = np.zeros((self.board_rep.shape[0], self.board_rep.shape[
            1]))  # board matrix is the abstract representation of the board
        self.numofwin = numofwin  # represents the number of the winning sequence, for tic-tac-toe it is 3 (3 in a row)
        self.numberof_steps = 0  # how many steps were made in a given game
        self.win = np.ones(self.numofwin), -np.ones(self.numofwin)  # these are the winning combinations in the board:
        #   for example 3 x or 3 o (3 1s, or 3 -1s in the abstract rep)
        self.neighbours = ((1, 0), (0, -1), (1, -1), (-1, -1))
        # this defines the relaitve neighbours of a square, here every square around a given square counts
        # as a neighbour. but because we will iterate form the left upmost square, we only need to check these 4
        # neighbours to realize a winning or losing position

    # refresh the boardmatrix with -1 or 1, depending on number of steps
    def refresh_boardmatrix(self, x_coord, y_coord):
        temp = self.which_element(x_coord, y_coord)
        i = temp[0]
        ii = temp[1]
        if self.numberof_steps % 2 is 0:
            self.board_matrix[i, ii] = -1
        else:
            self.board_matrix[i, ii] = 1

    # this is the function which checks if there is any winning sequence in the board. it checks for every square,
    # not just for the square where the last move were made, for minor speed advantage it can be rewritten
    # to only check for the last move, but for tic-tac-toe, it does not matter.
    # in this form it acts like a general search algorithm for words in a character matrix where the words can be
    # on diagonals also
    # it iterates over the winning sequences, in this case its 2 iteration one for "-1,-1,-1", one for "1,1,1"
    # the function searches for multiple occurrences of a sequence in a matrix, which in this game, doesn't make sense
    # but if there would be multiple occurrences, the positions of the starting point of the sequences will be
    # contained in a list
    def searchfor_word(self, word_):
        pos = list()
        # lets say the word_ is "xxx"
        # first we loop over the elements of the board
        for i, row in enumerate(self.board_matrix):
            for ii, element in enumerate(row):
                if element == word_[0]:  # if the element of the board is "x"
                    for j in self.neighbours:  # for every direction defined by neigbours
                        game_over = True  # we assume the game is over
                        for jj in range(self.numofwin):  # we check in every direction if there are 3 x-s
                            try:
                                # if we find a value not "x" in a given direction, the game is not over,
                                #  negative indexing is bad here, hence the if statment
                            
                                if word_[jj] != self.board_matrix[i + jj * j[0], ii + jj * j[1]] \
                                        or i + jj * j[0] < 0 or ii + jj * j[1] < 0:
                                    game_over = False
                                    break
                            except IndexError:  # if we go out of bounds of the board we break
                                game_over = False
                                break
                        if game_over is True:  # if cucc stayed 5, someone won the game
                            pos.append([i, ii])
                            break
        return pos

    # is the player winning? player 1 is represented as 1, player2 is represented as -1
    def is_winning(self, player):
        wordpositions = list()
        if player == 1:
            wordpositions.append(self.searchfor_word(self.win[0]))
        else:
            wordpositions.append(self.searchfor_word(self.win[1]))
        for i in wordpositions:
            if len(i) is not 0:
                return True
        return False

    # is the player losing?
    def is_losing(self, player):
        wordpositions = list()
        if player == 1:
            wordpositions.append(self.searchfor_word(self.win[1]))
        else:
            wordpositions.append(self.searchfor_word(self.win[0]))
        for i in wordpositions:
            if len(i) is not 0:
                return True
        return False

    # is it a draw?
    def is_draw(self):
        if 0 not in self.board_matrix:
            return True
        return False


# some hyperparameters
GAMMA = 0.5 #for exp. moving average
ALPHA = 0.0005 #learning rate


class Node:
    """Board states are stored in a hierarchical tree structure, allowing for tree search algorithms for more
     complex problems later."""

    def __init__(self, state, parent):
        self.parent = parent  # parent is the state, this state is originated from
        self.child = []  # this will store states originating from this state
        self.state = state  # this state
        self.times_visited = 0  # counts how many times this state was visited during training

    def add_child(self, child):
        self.child.append(child)


class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for a given state, and all actions (all possible states reacheable from this state for tic-tac-toe
    9 states, note that the network has to learn to recognize illegal moves, for example not choosing an action, where
    the wquare is already occupied)
    """

    # the state is the abstract representation of the board, but the state given to the network conists of this state,
    # and another whith the same dimension, which is 1 where a square is occupied and zero where it is not.
    # this extra representation may help to better recognize illegal moves
    # state_row : number of rows, state_col: number of columns
    def __init__(self, state_row, state_col, hidden_size=200):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_row * state_col * 2, hidden_size)  # *2 because the extra state
        self.state_encoder2 = nn.Linear(hidden_size, 100)
        # self.state2action = nn.Linear(hidden_size, action_dim*object_dim)
        self.state2coord = nn.Linear(100, state_row * state_col)

    def forward(self, x):
        state = F.leaky_relu(self.state_encoder(x))
        state = F.leaky_relu(self.state_encoder2(state))
        # return self.state2action(state)
        return self.state2coord(state)


class AI(AbstractRepresentation):
    """ The class of the AI. Since its a multiplayer, there will be two AIs playing each other when training"""

    def __init__(self, epsilon, player, state_dim, radius, resolution, square_num, numofwin):
        super().__init__(resolution, square_num, numofwin)
        self.player = player  # -1 or 1 depending if its the first player ai or the second
        self.end = False  # is the game over?
        self.dqn = DQN(*state_dim)  # q network
        self.dqn2 = DQN(*state_dim)  # second q network for double q learning algorithm
        self.radius = radius  # radius for the random selection method
        self.epsilon = epsilon  # epsilon for epsilon greedy policy
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=ALPHA)  # optimizer

    #  this method selects an empty square randomly from empty squares which have a non-empty square in its radius
    #  (self.radius) for example in gomoku where the board is huge, there is no point to randomly select any square,
    #  only squares which are close to where the actual game is happening, in tic tac toe it doesnt matter, so it
    #  works with radius = 2, which allows for selecting any empty squares in the board
    #  basically it collects where the empty sqares are in the board and pads the board with the radius, and checks
    #  that in the radius of the empty squares the sum of the abs of all square are zero or bigger than zero
    #  if its bigger than zero, there is a non-empty square in the radius, so we can choose this square
    #  padding is needed to not go out of bounds when we check the radius
    @staticmethod
    def rand_select(state, radius):
        # if all squares are empty we just randomly choose one
        if np.sum(np.abs(state)) == 0:
            return np.random.randint(0, state.shape[0], None), np.random.randint(0, state.shape[1], None)
        where_empty = np.argwhere(state == 0)
        padded = np.pad(state, radius, mode="constant", constant_values=0)
        where_empty += radius  # adjusting indecies for padded board
        indecies = list()
        for i, row in enumerate(where_empty):
            if np.sum(np.abs(
                    padded[(row[0] - radius):(row[0] + radius + 1), (row[1] - radius):(row[1] + radius + 1)])) > 0:
                indecies.append((row[0], row[1]))
        good_indecies = indecies[np.random.randint(0, len(indecies), None)]
        returned = good_indecies[0] - radius, good_indecies[1] - radius
        # print(returned)
        return returned

    #  this method adds the extra state to the state for the neural network input (extra state represents valid
    #  and invalid moves)
    @staticmethod
    def plusdimension(input):
        out = np.zeros((input.shape[0], input.shape[1], 2))
        out[:, :, 1][np.where(input == 0)] = 1
        out[:, :, 0] = input.copy()
        return out

    def epsilon_greedy(self, node):
        # epsilon-greedy policy. with probability of epsilon we choose a non-random move
        bin = np.random.binomial(1, self.epsilon, None)

        if bin == 1:

            newq = self.dqn.forward(torch.FloatTensor(self.plusdimension(node.state)).flatten()).detach().numpy()
            newq = np.reshape(newq, (self.square_num_row, self.square_num_col))
            # newq[np.where(self.board_matrix != 0)] = newq.min()-1  # for playing humans this should be uncommented
            # to ensure no illegal moves
            newcoord = np.unravel_index(newq.argmax(), (self.square_num_row,
                                                        self.square_num_col))  # gives the new state with the biggest q value (the coordinate where the player should )
            newstate = node.state.copy()
            newstate[newcoord] = self.player  # the new state with the new coordinate
            #  if the new state is already in one of the child nodes, we just add one to the child node's times_visited
            #  if its a new state we add a new child node. we return the child node and the new coordinates
            for nod in node.child:
                # print("here")
                if np.sum(np.equal(nod.state, newstate)) == nod.state.shape[0] * nod.state.shape[1]:
                    nod.times_visited += 1
                    return nod, newcoord

            child_node = Node(newstate, node)
            node.add_child(child_node)
            return child_node, newcoord

        # random selection
        if bin == 0:
            newcoord = self.rand_select(node.state, self.radius)
            newstate = node.state.copy()
            newstate[newcoord] = self.player
            for nod in node.child:
                if np.sum(np.equal(nod.state, newstate)) == nod.state.shape[0] * nod.state.shape[1]:
                    nod.times_visited += 1
                    return nod, newcoord

            child_node = Node(newstate, node)
            node.add_child(child_node)
            return child_node, newcoord

    #  implementing one epoch of q learning: the neural network part
    def epoch(self, node, reward, rowindex, colindex, new_node=None):
        if new_node is None:
            q_now = self.dqn(torch.FloatTensor(self.plusdimension(node.state)).flatten())
            q_now = q_now[rowindex * self.square_num_row + colindex]  # the output of the network is a flatenned 3*3
            # array for tic tac toe, so we have to index it the right way

            loss = 1 / 2 * (reward - q_now) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:

            with torch.no_grad():
                # double q learning, target (reward+Gamma*maxq_next) uses the second networks output, which is only
                # updated sometimes allowing for more stationary target
                maxq_next = self.dqn2(torch.FloatTensor(self.plusdimension(new_node.state)).flatten())
            maxq_next = maxq_next.max()
            q_now = self.dqn(torch.FloatTensor(self.plusdimension(node.state)).flatten())
            q_now = q_now[rowindex * self.square_num_row + colindex]

            loss = 1 / 2 * (reward + GAMMA * maxq_next - q_now) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # implementing one epoch of q learning: the rewards and conditions
    def runone(self, starting_node, for_training, ai_other=None):

        # this part is not for training, its for playing a human, to know when the game is over, if the human wins, or draws the game
        self.board_matrix = starting_node.state
        if self.is_losing(self.player):
            reward = -1
            self.end = True
            if for_training:
                return starting_node, None
            return starting_node, reward
        elif self.is_draw():
            reward = 0
            self.end = True
            if for_training:
                return starting_node, None
            return starting_node, reward

        # the actual learning part, first choosing an action (a new state) by greedy policy
        new_node, new_coord = self.epsilon_greedy(starting_node)
        self.board_matrix = new_node.state
        # if the board is empty, we just choose an action, no training
        if starting_node.parent is None:
            if for_training:
                return new_node, None
            return new_node, 0

        # if we choose an illegal action a big negative reward is given, the game is over
        elif np.sum(np.abs(starting_node.state)) == np.sum(np.abs(new_node.state)):
            reward = -5
            self.end = True
            if for_training:
                self.epoch(starting_node, reward, new_coord[0], new_coord[1])
                return new_node, None
            return new_node, reward
        # if the action wins the game we get a reward, and the other ai gets a minus reward.
        # we update both ai's neural networks (the other ai only now knows he lost, so we can only update now)
        elif self.is_winning(self.player):
            reward = 1
            self.end = True
            if for_training:
                coord = np.where(starting_node.parent.state - starting_node.state != 0)
                ai_other.epoch(starting_node.parent, -reward, coord[0], coord[1])
                self.epoch(starting_node, reward, new_coord[0], new_coord[1])
                return new_node, None
            return new_node, reward
        # same stuff, drawing a game has 0 reward for both ais
        elif self.is_draw():
            reward = 0
            self.end = True
            if for_training:
                coord = np.where(starting_node.parent.state - starting_node.state != 0)
                ai_other.epoch(starting_node.parent, reward, coord[0], coord[1])
                self.epoch(starting_node, reward, new_coord[0], new_coord[1])
                return new_node, None
            return new_node, reward
            # if for_training:
            #     self.other_lose(ai_other, False, starting_node)

        # if the game is not over we update the q-values of only the other ai.
        # This is because only now knows the other ai, what its next state will be. Since it is a multiplayer the next state
        # for a given ai is always 2 steps away from the current state, not 1 step like in a single palyer game
        else:
            reward = 0
            if for_training:
                coord = np.where(starting_node.parent.state - starting_node.state != 0)
                ai_other.epoch(starting_node.parent, reward, coord[0], coord[1], new_node)
                return new_node, None
            return new_node, reward


def AImain():
    starting_node = Node(ai1.board_matrix, None)
    training1, training2 = True, True
    for i in range(150000):
        ai1.end, ai2.end = False, False
        print(i)
        #  double q learning: the second network which computes the maxq for the target, only updates at every
        #  500. iteration, allowing for a more stationary target
        if i % 500 == 0:
            for target_param, local_param in zip(ai1.dqn2.parameters(),
                                                 ai1.dqn.parameters()):
                target_param.data.copy_(local_param.data)
            for target_param, local_param in zip(ai2.dqn2.parameters(),
                                                 ai2.dqn.parameters()):
                target_param.data.copy_(local_param.data)
        #  training: in one episode only one of the ai is trained, epsilon is gradually bigger, to allow more
        #  exploitation in the later stages of training
        if i < 1000:
            training1 = True
            training2 = False
            ai1.epsilon = 0.5
            ai2.epsilon = 0
        elif i < 2000:
            training1 = False
            training2 = True
            ai1.epsilon = 0
            ai2.epsilon = 0.5
        elif i % 4 == 0:
            training1 = True
            training2 = False
            ai1.epsilon = i / 150000
            ai2.epsilon = 0
        elif i % 4 == 1:
            training1 = True
            training2 = False
            ai1.epsilon = i / 150000
            ai2.epsilon = 0.8
        elif i % 4 == 2:
            training1 = False
            training2 = True
            ai1.epsilon = 0
            ai2.epsilon = i / 150000
        elif i % 4 == 3:
            training1 = False
            training2 = True
            ai1.epsilon = 0.8
            ai2.epsilon = i / 150000

        #  going back to the root of the tree after one episode
        while starting_node.parent is not None:
            starting_node = starting_node.parent
        #  training one episode
        while True:
            newnode = ai1.runone(starting_node, training1, ai2)[0]
            if ai1.end is True or ai2.end is True:
                break
            starting_node = ai2.runone(newnode, training2, ai1)[0]
            if ai2.end is True or ai1.end is True:
                break
    #  this is for testing purpuses
    for i in range(1000):
        ai1.end, ai2.end = False, False
        ai2.epsilon = 0
        ai1.epsilon = 1
        while starting_node.parent is not None:
            starting_node = starting_node.parent
        while True:
            newnode, reward1 = ai1.runone(starting_node, False, ai2)
            if ai1.end is True or ai2.end is True:
                # if reward1 == -5:
                #     print("shit")
                break
            starting_node, reward2 = ai2.runone(newnode, False, ai1)
            if ai2.end is True or ai1.end is True:
                break
        print(reward1, reward2)


pygame.init()

x = 1000  # int(input())
y = 1000  # int(input())
density = 3  # int(input())
resolution = (x, y)
c = AbstractRepresentation(resolution, density, 3)  # creating a general object of representing the board
ai1 = AI(0, 1, (density, density), 2, resolution, density,
         3)  # the ai object will contain its own representation, what he sees from the actual representation
ai2 = AI(0, -1, (density, density), 2, resolution, density,
         3)  # the ai object will contain its own representation, what he sees from the actual representation


AImain()
#
torch.save(ai1.dqn, "ainew")  # saving a network configuration after training
torch.save(ai2.dqn, "ainew")  # saving a network configuration after training

#  the main fucntion, we can play with the ai, and it also the function, where we draw the board,
#  and implement the game loop and drawing functions using the pygame library
def main():
    global displaysurf  # the surface of the game
    end = False  # is the game over?
    print("Do you want to start (Yes/No)?")
    ai_start = input()
    if ai_start == "No":
        starting_node = Node(c.board_matrix, None)
        aiturn = True
        ai1.end = False
        ai1.epsilon = 1
        ai1.dqn = torch.load('ai100') #loading an existing nn structure
        ai1.dqn.eval()
        ai = ai1
    elif ai_start == "Yes":
        newnode = Node(c.board_matrix, None)
        aiturn = False
        ai2.end = False
        ai2.epsilon = 1
        ai2.dqn = torch.load('ai100')
        ai2.dqn.eval()
        ai = ai2
    else:
        print("Wtf")
        return 0
    displaysurf = pygame.display.set_mode(resolution)

    while True:  # game loop
        drawlines()

        if aiturn is True:
            if ai.end is False:
                newnode, fik = ai.runone(starting_node, for_training=False)
                print(fik)
                #  clumsy way of updating the physical surface, depending on the new coordinate in the abstract space:
                if (np.sum(newnode.state - c.board_matrix) == 0):  # when the human ended the game there is no update
                    aiturn = False
                else:
                    pos = np.where(newnode.state - c.board_matrix != 0)
                    end = updateboard(
                        (c.board_rep[pos[0], pos[1], 0] + 0.00001, c.board_rep[pos[0], pos[1], 1] + 0.00001))
                    aiturn = False

        for event in pygame.event.get():
            if event.type is QUIT:  # handling quit
                pygame.quit()
                sys.exit()
            if event.type is pygame.MOUSEBUTTONUP:  # handling mouse button push
                if end is True:  # if the game is over break
                    break
                pos = pygame.mouse.get_pos()  # getting the position
                end = updateboard(pos)  # calling to updateboard
                starting_node = Node(c.board_matrix, newnode)
                aiturn = True

        pygame.display.update()


def drawlines():  # drawing the lines to separate the surface into squares
    for i, sub_representation in enumerate(c.board_rep):
        if i is 0:
            continue
        pygame.draw.aaline(displaysurf, (0, 0, 180), sub_representation[0, :],
                           (sub_representation[0, 0], resolution[1]))
    for i, sub_representation in enumerate(c.board_rep[0, :, :]):
        if i is 0:
            continue
        pygame.draw.aaline(displaysurf, (0, 0, 180), sub_representation, (resolution[0], sub_representation[1]))


def drawx(x_board, y_board, color):  # draws an x into a square depending of the coordinates
    border = int(c.length / 10)
    line1_beg = (c.board_rep[x_board, y_board][0] + border, c.board_rep[x_board, y_board][1] + border)
    line1_end = (
        c.board_rep[x_board, y_board][0] + c.length - border, c.board_rep[x_board, y_board][1] + c.length - border)
    line2_beg = (c.board_rep[x_board, y_board][0] + border, c.board_rep[x_board, y_board][1] + c.length - border)
    line2_end = (c.board_rep[x_board, y_board][0] + c.length - border, c.board_rep[x_board, y_board][1] + border)
    pygame.draw.aaline(displaysurf, color, line1_beg, line1_end)
    pygame.draw.aaline(displaysurf, color, line2_beg, line2_end)


def drawcircle(x_board, y_board, color):  # draws an o into a square depending of the coordinates
    border = int(c.length / 10)
    radius = int(c.length / 2 - border)
    circle_coord = (c.board_rep[x_board, y_board][0] + int(c.length / 2), c.board_rep[x_board, y_board][1] +
                    int(c.length / 2))
    pygame.gfxdraw.aacircle(displaysurf, circle_coord[0], circle_coord[1], radius, color)


def updateboard(pos):  # this functions calls to the draw functions and updates the abstract_representation
    # of the board. pos is the position of the cursor, when there is a mouse button action
    whichelement = c.which_element(pos[0], pos[1])
    defaultcolor1 = (100, 80, 50)
    defaultcolor2 = (50, 100, 120)
    winningcolor = (150, 0, 0)
    # if the square already contains a value "x", or "o" we finish
    if c.board_matrix[whichelement] in (-1, 1):
        return False
    if c.numberof_steps % 2 is 0:  # updating with "o"
        drawcircle(whichelement[0], whichelement[1], defaultcolor1)
        c.board_matrix[whichelement] = 1
    else:  # updating with "x"
        drawx(whichelement[0], whichelement[1], defaultcolor2)
        c.board_matrix[whichelement] = -1
    if c.is_winning(c.numberof_steps % 2) is True or c.is_draw() is True or \
            c.is_losing(
                c.numberof_steps % 2) is True:  # if there is a winning sequence we draw the shape with a different color, and return
        # true
        c.winning_rectangle = c.board_rep[whichelement]
        if c.numberof_steps % 2 is 0:
            drawcircle(whichelement[0], whichelement[1], winningcolor)
        else:
            drawx(whichelement[0], whichelement[1], winningcolor)
        return True
    c.numberof_steps += 1
    return False


main()
