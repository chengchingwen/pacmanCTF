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
from qlearningAgents import ApproximateQAgent


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, thirdIndex, isRed,
               first = 'myAgent', second = 'myAgent', third = 'myAgent'):
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



class myAgent(ApproximateQAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
    def getAction(self, gameState):
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
        successor = gameState.generateSuccessor(self.index, action)
        return successor
    def getFeatures(self, state, action):
        features = util.Counter()
        # extract the grid of food and wall locations and get the ghost locations
        Food = list(self.getFood(state))
        DefFood = list(self.getFoodYouAreDefending(state))
        walls = state.getWalls()
        myState = state.getAgentState(self.index)
        successor = state.generateSuccessor(self.index, action)
        newFood = list(self.getFood(successor))
        newDefFood = list(self.getFoodYouAreDefending(successor))
        newState = successor.getAgentState(self.index)
        newWalls = successor.getWalls()
        
        myPos = int(myState.getPosition()[0]), int(myState.getPosition()[1])
        newPos = int(newState.getPosition()[0]), int(newState.getPosition()[1])

        newGhostStates = [(successor.getAgentState(i),i)  for i in self.getOpponents(successor) if successor.getAgentState(i).configuration]
        newScaredTimes = [ghostState[0].scaredTimer for ghostState in newGhostStates]
        dangerzone = []
        scaredghost = []
        ghostpos = []

        for i in newGhostStates:
          if newState.isPacman or (not newState.isPacman and newState.scaredTimer) or newState.ownFlag or myState.ownFlag:
            if newScaredTimes[newGhostStates.index(i)]==0:
              ghostpos.append(i[0].getPosition())
              for state in [successor.generateSuccessor(i[1], way) for way in successor.getLegalActions(i[1])]:
                for state2 in [state.generateSuccessor(i[1], way2) for way2 in state.getLegalActions(i[1])]:
                  for state3 in [state2.generateSuccessor(i[1], way3) for way3 in state2.getLegalActions(i[1])]:
                    for a in [state3.getAgentState(m) for m in  self.getOpponents(state3) if successor.getAgentState(i[1]).configuration]:
                      if a.getPosition() and (not a.isPacman or (a.isPacman and (newState.ownFlag or myState.ownFlag))):
                        dangerzone.append(a.getPosition())
          else:
            if newScaredTimes[newGhostStates.index(i)]:
              scaredghost.append(i[0].getPosition())


        features["defending"] = float(not myState.isPacman)
        features["goAtteck"] = float(newState.isPacman)
        # compute the location of pacman after he takes the action
        features["bias"] = 1.0
        
        if myState.isPacman and newState.isPacman:
          if Food[newPos[0]][newPos[1]] ^ newFood[newPos[0]][newPos[1]]:
            features['eat-food'] +=1
        elif features["defending"] and features["goAttack"]:
          if Food[newPos[0]][newPos[1]] ^ newFood[newPos[0]][newPos[1]]:
            features['eat-food'] +=1
        elif features["defending"] and not newState.isPacman:
          for i in range(len(DefFood)):
            for j in range(len(DefFood[0])):
              if newDefFood[i][j]:
                features['defend-food'] +=1
        else:
          for i in range(len(DefFood)):
            for j in range(len(DefFood[0])):
              if newDefFood[i][j]:
                features['defend-food'] +=1
          
                           

        features["in-danger"] = float(newPos in dangerzone)
        
        features["own-flag"]=float(myState.ownFlag or newState.ownFlag)

        critical_index = self.getOwnFlagOpponent(successor)
        if critical_index:
          critical_pos = state3.getAgentState(critical_index).getPosution()
          if critical_pos:
            features["critical-point"] = self.getMazeDistance(newPos, critical_pos)
          else:
            features["critical-point"] = self.getMazeDistance(newPos, self.getFlagsYouAreDefending(successor))



        if (myState.isPacman and newState.isPacman) or (features["defending"] and features["goAttack"]):
          if scaredghost:
            features["eat-scared"] = min([self.getMazeDistance(newPos, pos) for pos in scaredghost])
        else:
          if ghostpos:
            features["eat-pacman"]= min([self.getMazeDistance(newPos, pos) for pos in ghostpos])



        if not myState.isPacman or not newState.isPacman:
          features['self-scared'] = myState.scaredTimer > 0


        dist = closestFood(newPos, Food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        return features

    def getQValue(self, state, action):
        return self.getWeights()*self.getFeatures(state, action)


    def update(self, state, action, nextState, reward):
        f = self.getFeatures(state, action)
        qmax = self.computeValueFromQValues(nextState)
        q = self.getQValue(state, action)
        for i in f.keys():
            self.weights[i] += self.alpha*(reward + self.discount * qmax - q) * f[i]




def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(int(pos[0]), int(pos[1]), 0)]
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
