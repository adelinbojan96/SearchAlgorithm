import util
import random

class SearchProblem:
    def getStartState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        util.raiseNotDefined()

def randomSearch(problem):

    current = problem.getStartState()
    solution = []

    while not problem.isGoalState(current):
        succ = problem.getSuccessors(current)
        no_of_successors = len(succ)

        if no_of_successors == 0:
            break

        random_succ_index = int(random.random() * no_of_successors)
        next = succ[random_succ_index]
        current = next[0]
        solution.append(next[1])

    print("The solution is", solution)
    return solution

def tinyMazeSearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):

    stack = util.Stack()
    start_state = problem.getStartState()
    stack.push((start_state, []))  # (state, path)


    visited = set()

    while not stack.isEmpty():
        state, path = stack.pop()


        if problem.isGoalState(state):
            return path


        if state not in visited:
            visited.add(state)

            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    stack.push((successor, path + [action]))

    return []


def breadthFirstSearch(problem):

    queue = util.Queue()
    start_state = problem.getStartState()
    queue.push((start_state, []))

    visited = set()

    while not queue.isEmpty():
        state, path = queue.pop()


        if problem.isGoalState(state):
            return path


        if state not in visited:
            visited.add(state)


            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    queue.push((successor, path + [action]))

    return []


def uniformCostSearch(problem):
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    util.raiseNotDefined()

rs = randomSearch
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
