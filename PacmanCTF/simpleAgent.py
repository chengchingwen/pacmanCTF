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
               first = 'myAgent', second = 'myDAgent', third = 'myAgent'):
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
        self.myFlagPos = self.getFlagsYouAreDefending(gameState)[0]
        self.depth = 1
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
        actions = gameState.getLegalActions(self.index)
      
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.getFeatures1(gameState, a)*self.getWeights1(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        legalActions = gameState.getLegalActions(self.index)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.value(gameState, self.index, -float("inf"), float("inf"))[1]
        self.doAction(gameState,action)
#        return random.choice(bestActions)
        return action
    def getFeatures1(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()    
        features['successorScore'] = -len(foodList)#self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
          myPos = successor.getAgentState(self.index).getPosition()
          minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
          features['distanceToFood'] = minDistance
        return features

    def getWeights1(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}


    def value(self, state, index, a, b):
        mindex=index%6
        if index/6 >= self.depth or state.isOver():
            z= (self.getValue( state), "Stop")
            return z
        if mindex in self.getTeam(state) and state.getAgentState(mindex).configuration:
            i=self.maxv(state, index, a, b)
            return i
        elif mindex in self.getOpponents(state) and state.getAgentState(mindex).configuration:
            j=random.choice((self.minv, self.expv))(state, index, a, b)
            return j
        else:
            return self.value(state, index+1, a, b)

    def maxv(self, state, index, a, b):
        v=(-float("inf"),'Stop')
        nindex=(index)%6
        for way in state.getLegalActions(nindex):
            k=(self.value(state.generateSuccessor(nindex, way),index+1, a ,b)[0],way)
            if k[0]>=v[0]:
                v=k
            if v[0]>b:
                return v
            if v[0]>=a:
                a=v[0]
        return v

    def minv(self, state, index, a, b):
        v=(float("inf"),'Stop')
        nindex=(index)%6
        for way in state.getLegalActions(nindex):
            k=(self.value(state.generateSuccessor(nindex, way),index+1, a, b)[0],way)
            if k[0]<=v[0]:
                v=k
            if v[0]<a:
                return v
            if v[0]<=b:
                b=v[0]
        return v
    def expv(self, state, index, a, b):
        v=0 
        nindex=(index)%6
        for way in state.getLegalActions(nindex):
            p=1.0/len(state.getLegalActions(nindex))
            k=(self.value(state.generateSuccessor(nindex, way),index+1, a, b)[0],way)
            v+=p * k[0]
        return (v, way)


    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        return successor
    def getFeatures(self, state, action):
        features = util.Counter()
        Food = list(self.getFood(state))
        DefFood = self.getFoodYouAreDefending(state).asList()
        walls = state.getWalls()
        myState = state.getAgentState(self.index)
        successor = state.generateSuccessor(self.index, action)
        newFood = list(self.getFood(successor))
        newDefFood = self.getFoodYouAreDefending(successor).asList()
        newState = successor.getAgentState(self.index)
        newWalls = successor.getWalls()
        features['successorScore'] = self.getScore(successor)
        foodList = self.getFood(successor).asList()    
        features['has-food'] = len(foodList)
        myPos = int(myState.getPosition()[0]), int(myState.getPosition()[1])
        newPos = int(newState.getPosition()[0]), int(newState.getPosition()[1])
        newGhostStates = [(successor.getAgentState(i),i)  for i in self.getOpponents(successor) if successor.getAgentState(i).configuration]
        newScaredTimes = [ghostState[0].scaredTimer for ghostState in newGhostStates]
        dangerzone = []
        scaredghost = []
        ghostpos = []
        if len(foodList) > 0:
          myPos = successor.getAgentState(self.index).getPosition()
          minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
          features['distanceToFood'] = minDistance

        for i in newGhostStates:
          if newState.isPacman or (not newState.isPacman and newState.scaredTimer) or newState.ownFlag or myState.ownFlag:
            if newScaredTimes[newGhostStates.index(i)]==0:
              ghostpos.append(i[0].getPosition())
              for state in [successor.generateSuccessor(i[1], way) for way in successor.getLegalActions(i[1])]:
                for a in [state.getAgentState(m) for m in  self.getOpponents(state) if state.getAgentState(i[1]).configuration]:
                  if a.getPosition() and (not a.isPacman or (a.isPacman and (newState.ownFlag or myState.ownFlag))):
                    dangerzone.append(a.getPosition())
          else:
            if newScaredTimes[newGhostStates.index(i)]:
              scaredghost.append(i[0].getPosition())

        if features["has-food"]:
          features["goAttack"] = 1

        features["in-danger"] = float(newPos in dangerzone)        
        features["own-flag"]=float(myState.ownFlag or newState.ownFlag)
        critical_index = self.getOwnFlagOpponent(state)
        if critical_index:
          critical_pos = state.getAgentState(critical_index).getPosition()
          if critical_pos:
            features["critical-point"] = self.getMazeDistance(newPos, critical_pos)
          else:
            features["critical-point"] = self.getMazeDistance(newPos, self.myFlagPos)

        features.divideAll(3.1415926**3.1415926**3.1415926)
        return features

    def getQValue(self, state, action):
        return self.getWeights()*self.getFeatures(state, action)


    def update(self, state, action, nextState, reward):
        f = self.getFeatures(state, action)
        qmax = self.computeValueFromQValues(nextState)
        q = self.getQValue(state, action)
        for i in f.keys():
            self.weights[i] += self.alpha*(reward + self.discount * qmax - q) * f[i]

class myDAgent(ApproximateQAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.myFlagPos = self.getFlagsYouAreDefending(gameState)[0]
        self.depth = 1
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
        actions = gameState.getLegalActions(self.index)
      
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.getFeatures1(gameState, a)*self.getWeights1(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        legalActions = gameState.getLegalActions(self.index)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.value(gameState, self.index, -float("inf"), float("inf"))[1]
        self.doAction(gameState,action)
#        return random.choice(bestActions)
        return action
    def getFeatures1(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
          features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights1(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


    def value(self, state, index, a, b):
        mindex=index%6
        if index/6 >= self.depth or state.isOver():
            z= (self.getValue( state), "Stop")
            return z
        if mindex in self.getTeam(state) and state.getAgentState(mindex).configuration:
            i=self.maxv(state, index, a, b)
            return i
        elif mindex in self.getOpponents(state) and state.getAgentState(mindex).configuration:
            j=random.choice((self.minv, self.expv))(state, index, a, b)
            return j
        else:
            return self.value(state, index+1, a, b)

    def maxv(self, state, index, a, b):
        v=(-float("inf"),'Stop')
        nindex=(index)%6
        for way in state.getLegalActions(nindex):
            k=(self.value(state.generateSuccessor(nindex, way),index+1, a ,b)[0],way)
            if k[0]>=v[0]:
                v=k
            if v[0]>b:
                return v
            if v[0]>=a:
                a=v[0]
        return v

    def minv(self, state, index, a, b):
        v=(float("inf"),'Stop')
        nindex=(index)%6
        for way in state.getLegalActions(nindex):
            k=(self.value(state.generateSuccessor(nindex, way),index+1, a, b)[0],way)
            if k[0]<=v[0]:
                v=k
            if v[0]<a:
                return v
            if v[0]<=b:
                b=v[0]
        return v
    def expv(self, state, index, a, b):
        v=0 
        nindex=(index)%6
        for way in state.getLegalActions(nindex):
            p=1.0/len(state.getLegalActions(nindex))
            k=(self.value(state.generateSuccessor(nindex, way),index+1, a, b)[0],way)
            v+=p * k[0]
        return (v, way)


    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        return successor
    def getFeatures(self, state, action):
        features = util.Counter()
        Food = list(self.getFood(state))
        DefFood = self.getFoodYouAreDefending(state).asList()
        walls = state.getWalls()
        myState = state.getAgentState(self.index)
        successor = state.generateSuccessor(self.index, action)
        newFood = list(self.getFood(successor))
        newDefFood = self.getFoodYouAreDefending(successor).asList()
        newState = successor.getAgentState(self.index)
        newWalls = successor.getWalls()
        features['successorScore'] = self.getScore(successor)
        foodList = self.getFood(successor).asList()    
        features['has-food'] = len(foodList)
        myPos = int(myState.getPosition()[0]), int(myState.getPosition()[1])
        newPos = int(newState.getPosition()[0]), int(newState.getPosition()[1])
        newGhostStates = [(successor.getAgentState(i),i)  for i in self.getOpponents(successor) if successor.getAgentState(i).configuration]
        newScaredTimes = [ghostState[0].scaredTimer for ghostState in newGhostStates]
        dangerzone = []
        scaredghost = []
        ghostpos = []
        if len(foodList) > 0:
          myPos = successor.getAgentState(self.index).getPosition()
          minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
          features['distanceToFood'] = minDistance

        for i in newGhostStates:
          if newState.isPacman or (not newState.isPacman and newState.scaredTimer) or newState.ownFlag or myState.ownFlag:
            if newScaredTimes[newGhostStates.index(i)]==0:
              ghostpos.append(i[0].getPosition())
              for state in [successor.generateSuccessor(i[1], way) for way in successor.getLegalActions(i[1])]:
                for a in [state.getAgentState(m) for m in  self.getOpponents(state) if state.getAgentState(i[1]).configuration]:
                  if a.getPosition() and (not a.isPacman or (a.isPacman and (newState.ownFlag or myState.ownFlag))):
                    dangerzone.append(a.getPosition())
          else:
            if newScaredTimes[newGhostStates.index(i)]:
              scaredghost.append(i[0].getPosition())

        if features["has-food"]:
          features["goAttack"] = 1
        
        if state.getAgentDistances():
          features["eat-pacman"] = -min([min(state.getAgentDistances()[i],successor.getAgentDistances()[i]) for i in self.getOpponents(successor)])          
        features["in-danger"] = float(newPos in dangerzone)        
        features["own-flag"]=float(myState.ownFlag or newState.ownFlag)
        critical_index = self.getOwnFlagOpponent(state)
        if critical_index:
          critical_pos = state.getAgentState(critical_index).getPosition()
          if critical_pos:
            features["critical-point"] = self.getMazeDistance(newPos, critical_pos)
          else:
            features["critical-point"] = self.getMazeDistance(newPos, self.myFlagPos)

        features.divideAll(3.1415926**3.1415926**3.1415926)
        return features

    def getQValue(self, state, action):
        return self.getWeights()*self.getFeatures(state, action)


    def update(self, state, action, nextState, reward):
        f = self.getFeatures(state, action)
        qmax = self.computeValueFromQValues(nextState)
        q = self.getQValue(state, action)
        for i in f.keys():
            self.weights[i] += self.alpha*(reward + self.discount * qmax - q) * f[i]



