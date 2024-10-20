from game import Directions
from util import manhattanDistance
import util, layout
import sys, types, time, random, os


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

def fitness(genome, problem):
    state = problem.getStartState()
    score = 0
    previous_food_count = state.getNumFood()
    steps_survived = 0

    for i in range(0, len(genome), 2):
        move = genome[i:i + 2]
        action = decodeMove(move)

        successors = problem.getSuccessors(state)
        for successor, direction, cost in successors:
            if direction == action:
                state = successor
                break

        steps_survived += 1

        if state.isWin():
            score += 500000
            break

        if state.isLose():
            score -= 50000
            break

        score += 20

        pacman_position = state.getPacmanPosition()
        ghosts = state.getGhostPositions()
        current_food_count = state.getNumFood()

        if current_food_count < previous_food_count:
            score += 100
            previous_food_count = current_food_count

        if ghosts:
            closest_ghost_distance = min(
                [manhattanDistance(pacman_position, ghost) for ghost in ghosts]
            )
            if closest_ghost_distance < 1:
                score -= 10000
            if closest_ghost_distance < 2:
                score -= 1000
            elif closest_ghost_distance < 3:
                score -= 500
            else:
                score += closest_ghost_distance * 10

    score += steps_survived * 10

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
    index1 = fitnessScores.index(max(fitnessScores))

    copyMax = max(fitnessScores)
    fitnessScores[index1] = float('-inf')

    index2 = fitnessScores.index(max(fitnessScores))

    fitnessScores[index1] = copyMax

    parent1 = population[index1]
    parent2 = population[index2]
    return parent1, parent2

def geneticAlgorithmSearch(problem, population_size=200, generations = 3, mutation_rate=0.75, max_genome_length=700):

    # generating the initial population of genomes
    randomGenomesPopulation = [generateGenome(max_genome_length) for _ in range(population_size)]

    fitnessScore = [fitness(genome, problem) for genome in randomGenomesPopulation]

    for i in range(generations):
        newPopulation = []

        while len(newPopulation) < population_size:
            parent1, parent2 = select(randomGenomesPopulation, fitnessScore)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            newPopulation.append(child1)
            if len(newPopulation) < population_size:
                newPopulation.append(child2)

        fitnessScore = [fitness(genome, problem) for genome in newPopulation]
        randomGenomesPopulation = newPopulation

        best_fitness = max(fitnessScore)
        avg_fitness = sum(fitnessScore) / len(fitnessScore)
        print(f"Generation {i + 1}, Best Fitness: {best_fitness}, Average Fitness: {avg_fitness}")

    bestGenomeIndex = fitnessScore.index(max(fitnessScore))
    bestGenome = randomGenomesPopulation[bestGenomeIndex]

    state = problem.getStartState()
    actions = []
    for i in range(0, len(bestGenome), 2):
        move = bestGenome[i:i + 2]
        action = decodeMove(move)
        legalActions = state.getLegalPacmanActions()
        if action in legalActions:
            actions.append(action)
            state = state.generatePacmanSuccessor(action)
            if state.isWin() or state.isLose():
                break
        else:
            if legalActions:
                action = random.choice(legalActions)
                actions.append(action)
                state = state.generatePacmanSuccessor(action)
            else:
                continue

    return actions


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
gas = geneticAlgorithmSearch
