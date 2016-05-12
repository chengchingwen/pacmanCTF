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

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)



class myAgent(ApproximateQAgent):
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

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = self.getFood(state)
        walls = state.getWalls()
        #ghosts = self.getGhostPositions()
        ghosts = [ state.getAgentPosition(x) for x in self.getOpponents(state)]
        features = util.Counter()
        successor = state.generateSuccessor(self.index, action)
        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts if g)
        if myState.isPacman:features['ghost-distance']*=-1

        features["own-flag"]=myState.ownFlag


        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            invdists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(invdists)
        scardemy = [i for i in enemies if not i.isPacman and i.scaredTimer]
        features['num-scard-emy']= len(scardemy)
        features['totalscared']=0
        if len(scardemy) > 0:
            scadists = [self.getMazeDistance(myPos, a.getPosition()) for a in scardemy]
            features['scaredDistance'] = min(scadists)
        oppcaps = self.getCapsules(successor)
        features['self-scared'] = myState.scaredTimer > 0
        features['num-of-caps'] = len(oppcaps)
        flagopp = self.getOwnFlagOpponent(successor)
        if flagopp:
            features['flagopp'] = self.getMazeDistance(myPos,successor.getAgentState(flagopp).getPosition())
        else:features['flagopp'] = 0
        if len(scardemy) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in scardemy]
            features['scaredDistance'] = min(dists)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
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