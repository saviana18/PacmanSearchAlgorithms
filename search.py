# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import random
import sys

w = 1000

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
    
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())"""

    startNode = problem.getStartState()
    if problem.isGoalState(startNode):
    	return []

    myStack = util.Stack()
    visitedNodes = []
    myStack.push((startNode, []))
    actions = []

    while not myStack.isEmpty():
    	currentNode, actions = myStack.pop()
    	if currentNode not in visitedNodes:
    		visitedNodes.append(currentNode)
    		if problem.isGoalState(currentNode):
    			return actions
    		for nextNode, action, cost in problem.getSuccessors(currentNode):
    			newAction = actions + [action]
    			myStack.push((nextNode, newAction))
    return actions
                
def randomSearch(problem):
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    currentState = problem.getStartState()
    actions = []
    while not (problem.isGoalState(currentState)):
    	succ = problem.getSuccessors(currentState)
    	randomNumber = random.randint(0, len(succ)-1)
    	next = succ[randomNumber]
    	currentState = next[0]
    	actions.append(next[1])
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    startNode = problem.getStartState()
    if problem.isGoalState(startNode):
    	return []
    
    myQueue = util.Queue()
    visitedNodes = []
    myQueue.push((startNode, []))
    
    while not myQueue.isEmpty():
    	currentNode, actions = myQueue.pop()
    	if currentNode not in visitedNodes:
    		visitedNodes.append(currentNode)
    		if problem.isGoalState(currentNode):
    			return actions
    		for nextNode, action, cost in problem.getSuccessors(currentNode):
    			newAction = actions + [action]
    			myQueue.push((nextNode, newAction))
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    startNode = problem.getStartState()
    if problem.isGoalState(startNode):
    	return []
    
    myPriorityQueue = util.PriorityQueue()
    visitedNodes = []
    myPriorityQueue.push((startNode, [], 0), 0)
    
    while not myPriorityQueue.isEmpty():
    	currentNode, actions, cost = myPriorityQueue.pop()
    	if currentNode not in visitedNodes:
    		visitedNodes.append(currentNode)
    		if problem.isGoalState(currentNode):
    			return actions
    		for nextNode, action, nextCost in problem.getSuccessors(currentNode):
    			newAction = actions + [action]
    			priority = cost + nextCost
    			myPriorityQueue.push((nextNode, newAction, priority), priority)
    			
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first.
    A* is an informed search algorithm, or a best-first search, meaning that it is formulated 
    in terms of weighted graphs: starting from a specific starting node of a graph, it aims
    to find a path to the given goal node having the smallest cost.
    On finite graphs with non-negative edge weights A* is guaranteed to terminate and is complete.
    Also, if the heuristic function used by A* is admissible, then A* is admissible.
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())"""
    
    startNode = problem.getStartState()
    myPriorityQueue = util.PriorityQueue()
    visitedNodes = []
    
    myPriorityQueue.push((startNode, [], 0), 0)
    if problem.isGoalState(startNode):
        return []

    while not myPriorityQueue.isEmpty():
        (currentNode, currentAction, currentCost) = myPriorityQueue.pop()
        if currentNode not in visitedNodes:
            visitedNodes.append(currentNode)
            if problem.isGoalState(currentNode):
                return currentAction
            for nextNode, nextAction, nextCost in problem.getSuccessors(currentNode):
                newAction = currentAction + [nextAction]
                newCost = currentCost + nextCost
                priority = newCost + heuristic(nextNode, problem)
                myPriorityQueue.push((nextNode, newAction, newCost), priority)
            
    util.raiseNotDefined()  
    
class NodeWithCost:
    def __init__(self, state, parent, action, cost):
    	self.state = state
    	self.parent = parent
    	self.action = action
    	self.cost = cost
    def getState(self): return self. state
    def getAction(self): return self.action
    def getParent(self): return self.parent
    def getCost(self): return self.cost

def weightedAStarSearch(problem, heuristic=nullHeuristic):
    """Weighted A*: expands states in the order of f = g+w*h values, w > 1 = bias 
    towards states that are closer to goal."""
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    
    startNode = problem.getStartState()
    myPriorityQueue = util.PriorityQueue()
    visitedNodes = []
    myPriorityQueue.push((startNode, [], 0), 0)

    while not myPriorityQueue.isEmpty():
        (currentNode, currentAction, currentCost) = myPriorityQueue.pop()
        if currentNode not in visitedNodes:
            visitedNodes.append(currentNode)
            if problem.isGoalState(currentNode):
                return currentAction
            for nextNode, nextAction, nextCost in problem.getSuccessors(currentNode):
                newAction = currentAction + [nextAction]
                newCost = currentCost + nextCost
                #w - the weight which is globally declared
                priority = newCost + w * heuristic(nextNode, problem)
                myPriorityQueue.push((nextNode, newAction, newCost), priority)
    
def weightedAStarSearchAux(problem, weight, heuristic=nullHeuristic):
    """This is an auxiliary function used for anytimeAStarSearch."""
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    
    startNode = problem.getStartState()
    myPriorityQueue = util.PriorityQueue()
    visitedNodes = []
    myPriorityQueue.push((startNode, [], 0), 0)

    while not myPriorityQueue.isEmpty():
        (currentNode, currentAction, currentCost) = myPriorityQueue.pop()
        if currentNode not in visitedNodes:
            visitedNodes.append(currentNode)
            if problem.isGoalState(currentNode):
                return weight, currentAction
            for nextNode, nextAction, nextCost in problem.getSuccessors(currentNode):
                newAction = currentAction + [nextAction]
                newCost = currentCost + nextCost
                priority = newCost + weight * heuristic(nextNode, problem)
                myPriorityQueue.push((nextNode, newAction, newCost), priority)
    return weight, currentAction
                
def anytimeAStarSearch(problem, heuristic=nullHeuristic):
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    
    done = False
    weight = w
    
    while True:
    	search, direction = weightedAStarSearchAux(problem, weight, heuristic=nullHeuristic)
    	print search
    	
    	if search > 1:
    		weight -= 100
    	else: 
    		return direction     

def idaSearch(startNode, myStack, threshold, visited, problem, heuristic):
    """This is an auxiliary function used for iterativeDeepeningAStarSearch."""
    
    if not myStack.isEmpty():
    	currentNodeWithCost = NodeWithCost(startNode.getState(), None, None, 0)
        currentNodeWithCost = myStack.pop()

    estCost = currentNodeWithCost.getCost() + heuristic(currentNodeWithCost.getState(), problem)
    actions = []
    if estCost > threshold:
        return estCost, actions
        
    if problem.isGoalState(currentNodeWithCost.getState()):
        return True, actions
        
    visited.append(currentNodeWithCost.getState())
    actions.append(currentNodeWithCost.getAction())
    
    minValue = sys.maxint
    
    for nextNode in problem.getSuccessors(currentNodeWithCost.getState()):
        if nextNode[0] not in visited:
            	updatedCostNode = NodeWithCost(nextNode[0], currentNodeWithCost, nextNode[1], nextNode[2] + currentNodeWithCost.getCost())
            	myStack.push(updatedCostNode)		
            	search, actions = idaSearch(startNode, myStack, threshold, visited, problem, heuristic)
            	
            	if(search == True):
                	return True, actions
            	if (search < minValue):
                	minValue = search
            	if not myStack.isEmpty():
            		currentNodeWithCost = myStack.pop()
    return minValue, actions

def iterativeDeepeningAStarSearch(problem, heuristic=nullHeuristic):
    """Iterative-deepening-A* works as follows: at each iteration, perform a depth-first search, cutting off a branch when
    its total cost f(n)=g(n)+h(n) exceeds a given threshold. This threshold starts at the estimate of the cost at the
    initial state, and increases for each iteration of the algorithm. At each iteration, the threshold used for the next
    iteration is the minimum cost of all values that exceeded the current threshold."""
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    startNode = NodeWithCost(problem.getStartState(), None, None, 0)
    threshold = heuristic(startNode.getState(), problem)
    found = False
    myStack = util.Stack()
    
    while True:
        visited = []
        myStack.push(startNode)
        search, actions = idaSearch(startNode, myStack, threshold, visited, problem, heuristic)
        if (search == True):
        	found = True
        	return actions
        if (search == sys.maxint):
        	found = False
            	return found
        threshold = search

def depthLimitedSearch(problem, depth):
    startNode = problem.getStartState()
    myStack = util.Stack()
    visitedNodes = []
    myStack.push((startNode, [], 0))
  
    while not myStack.isEmpty():
    	current = myStack.pop()
    	if current[0] not in visitedNodes:
    		visitedNodes.append(current[0])
    		if problem.isGoalState(current[0]):
    			return current[1]
    		if current[2] > depth:
    			return []
    		else:	
    			for nextState, nextAction, nextCost in problem.getSuccessors(current[0]):
    				newAction = current[1] + [nextAction]
    				newCost = current[2] + nextCost
    				myStack.push((nextState, newAction, newCost))

def iterativeDeepeningSearch(problem):
    """"IDS combines depth-first search's space-efficiency and breadth-first search's completeness (when the branching
    factor is finite). If a solution exists, it will find a solution path with the fewest arcs.
    Since iterative deepening visits states multiple times, it may seem wasteful, but it turns out to be not so costly,
    since in a tree most of the nodes are in the bottom level, so it does not matter much if the upper levels are visited
    multiple times."""
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    limit = 0
    while True:
        searchList = depthLimitedSearch(problem, limit)
        if searchList:
            return searchList
        else:
            limit += 1
    		
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
rs = randomSearch
wastar = weightedAStarSearch
atastar = anytimeAStarSearch
ida = iterativeDeepeningAStarSearch
ids = iterativeDeepeningSearch
