# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from multiAgents import AlphaBetaAgent,  ExpectimaxAgent


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, thirdIndex, isRed,
               first = 'expAgent', second = 'alphaAgent', third = 'expAgent'):
  """
  This function should return a list of three agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  print firstIndex, secondIndex, thirdIndex
  return [eval(first)(firstIndex), eval(second)(secondIndex), eval(third)(thirdIndex)]

##########
# Agents #
##########



class expAgent( ExpectimaxAgent):
    '''def __init__(self, index):
          ApproximateQAgent.__init__(self, index)

        print index, self
        Capt    ureAgent.__init__(self, index)
        print self.index
    '''
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def getAction(self, gameState):
        """
        Calls chooseAction on a grid position, but continues on half positions.
        If you subclass CaptureAgent, you shouldn't need to override this method.  It
        takes care of appending the current gameState on to your observation history
        (so you have a record of the game states of the game) and will call your
        choose action method if you're in a state (rather than halfway through your last
        move - this occurs because Pacman agents move half as quickly as ghost agents).

        """
        self.observationHistory.append(gameState)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myPos != util.nearestPoint(myPos):
          # We're halfway from one position to the next
              return gameState.getLegalActions(self.index)[0]
        else:
              return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        return self.GetAction(gameState)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor


class alphaAgent(AlphaBetaAgent):
    '''def __init__(self, index):
          ApproximateQAgent.__init__(self, index)

        print index, self
        Capt    ureAgent.__init__(self, index)
        print self.index
    '''
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def getAction(self, gameState):
        """
        Calls chooseAction on a grid position, but continues on half positions.
        If you subclass CaptureAgent, you shouldn't need to override this method.  It
        takes care of appending the current gameState on to your observation history
        (so you have a record of the game states of the game) and will call your
        choose action method if you're in a state (rather than halfway through your last
        move - this occurs because Pacman agents move half as quickly as ghost agents).

        """
        self.observationHistory.append(gameState)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myPos != util.nearestPoint(myPos):
          # We're halfway from one position to the next
              return gameState.getLegalActions(self.index)[0]
        else:
              return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        return self.GetAction(gameState)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor



def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

