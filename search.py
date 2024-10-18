import util
import random

from pac3man.reinforcement.game import Directions


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

def generateGenome(length):
    genome = ''.join([random.choice(['00', '01', '10', '11']) for _ in range(length)])
    return genome

def decodeMove(move):
    if move == "00":
        return Directions.NORTH
    elif move == "01":
        return Directions.EAST
    elif move == "10":
        return Directions.SOUTH
    else:
        return Directions.WEST

def getGhostPositions(state):
    ghostPositions = []
    for i in range(1, state.getNumAgents()):  # pacman is agent 0, ghosts are 1 through n
        ghostPositions.append(state.getGhostPosition(i))
    return ghostPositions

def manhattanDistance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def fitness(genome, problem):
    state = problem.getStartState
    ghosts = getGhostPositions(state)
    score = 0

    for i in range(0, len(genome), 2):
        move = genome[i:i+2]
        action = decodeMove(move)

        successors = problem.getSuccessors(state)
        validMove = False
        for successor, direction, cost in successors:
            if direction == action:
                state = successor
                validMove = True
                break

        if not validMove:
            return -999999

        closest_ghost_distance = min([manhattanDistance(state, ghost) for ghost in ghosts])
        score += closest_ghost_distance  # the farther, the better

    return score

def crossover(parent1, parent2):
    crossoverPoint = random.randint(2, len(parent1) - 2)

    child1 = parent1[:crossoverPoint] + parent2[crossoverPoint:]
    child2 = parent2[:crossoverPoint] + parent1[crossoverPoint:]

    return child1, child2

def chooseNewMove(possibleMoves, encodedMove):
    while True:
        newMove = possibleMoves[random.randint(0, len(possibleMoves) - 1)]

        if newMove != encodedMove:
            break

    return newMove

def mutate(genome, mutationRate):
    possibleMoves = ['00', '01', '10', '11']
    mutatedGenome = ""

    for i in range(0, len(genome), 2):
        encodedMove = genome[i:i+2]

        if random.random() < mutationRate:
            newMove = chooseNewMove(possibleMoves, encodedMove)
            mutatedGenome += newMove
        else:
            mutatedGenome += encodedMove
    return mutatedGenome

def select(population, fitnessScores):
    totalFitness = sum(fitnessScores)
    selectionProbs = [score / totalFitness for score in fitnessScores]
    maxValue = 0
    maxSelectedScoreIndex = 0

    for i in range(0, len(selectionProbs), 1):
        if selectionProbs[i] > maxValue:
            maxValue = selectionProbs[i]
            maxSelectedScoreIndex = i

    return population[maxSelectedScoreIndex]

def geneticAlgorithmSearch(problem, population_size=100, generations=200, mutation_rate=0.1, max_genome_length=50):
    
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

def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5


def aStarSearch(problem, heuristic):

    start_state = problem.getStartState()

    if problem.isGoalState(start_state):
        return []

    open_list = util.PriorityQueue()

    g_costs = {}
    g_costs[start_state] = 0


    open_list.push((start_state, []), heuristic(start_state, problem))


    closed_list = set()

    while not open_list.isEmpty():
        current_state, path = open_list.pop()


        if problem.isGoalState(current_state):
            return path


        if current_state in closed_list:
            continue


        closed_list.add(current_state)


        for successor, action, step_cost in problem.getSuccessors(current_state):
            new_path = path + [action]
            new_g_cost = g_costs[current_state] + step_cost
            new_f_cost = new_g_cost + heuristic(successor, problem)


            if successor not in g_costs or new_g_cost < g_costs[successor]:
                g_costs[successor] = new_g_cost
                open_list.push((successor, new_path), new_f_cost)

    return []


def uniformCostSearch(problem):
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    return 0


rs = randomSearch
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
