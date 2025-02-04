import random
import numpy as np
import copy                              

START = (0, 0)  # Starting position (x, y)
GOAL = (36, 12)

POPUL_Size = 150  
Mutation_rate = 1.0
Initial_chromosome_length = 15
Generations = 40000
Gen_Mat = 'DURL'  # all possible directions D: Down  U: Up  R: Right  L: Left

stagnation_mutation_change = 50   # threshold of number of generations without fitness improvement, for triggering a mutation change
penalty_value = 100    
pond_dist_goal = 2.9  
pond_valid_steps = 3.0  
generations_to_increase_chromosome = 100  



def load_maze_from_file(file_path):
    ''' 
    function that reads the maze from a txt file, and creates a 2D matrix
    '''
    maze = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.split() # split() splits the line into a list of strings based on whitespace
            values_line = []
            for num in line:
                clean_num = num.rstrip(',').rstrip(']')  # remove the comma and the last ] of the individual values 
                
                if clean_num.isdigit():
                    values_line.append(int(clean_num))  # adding the number as an integer to its row
            maze.append(values_line)
    return maze


def print_maze(maze, path):
    maze_copy = copy.deepcopy(maze)
    
    # mark the path in the maze
    for i, pos in enumerate(path):
        x, y = pos
        maze_copy[y][x] = "*"  # mark the path with *

    # print the maze with formatting (4 characters wide for alignment)
    for row in maze_copy:
        print(' '.join(f'{str(num):4}' for num in row))  # ensure alignment, '*' will also fit in 4 characters


def print_initial_maze(maze):
    maze_copy = copy.deepcopy(maze)
    
    # mark the start and goal
    maze_copy[START[1]][START[0]] = "S"  
    maze_copy[GOAL[1]][GOAL[0]] = "G" 

    for row in maze_copy:
        print(' '.join(f'{str(num):4}' for num in row))  # Ensure alignment, '*' will also fit in 4 characters
    print()


def target_reached(pos_mouse, goal):
    '''
    function that returns true, if the position of the mouse is the same as the goal 
    '''
    return pos_mouse == goal


def possible_move(maze, dest):
    ''' 
    returns true if the move to the destination is possible (no wall, or out of bounds)
    '''
    # check x coordinate iniside maze
    if dest[0] >= len(maze[0]) or dest[0] < 0:
        return False
    # check y coordinate inide maze
    elif dest[1] >= len(maze) or dest[1] < 0:
        return False
    # check wall
    elif maze[dest[1]][dest[0]] == 100:
        return False
    
    else: return True 


def move(pos, direction):
    '''
    given the current position and the direction it wants to move, it returns the new position after the move 
    '''
    x = pos[0]
    y = pos[1]
    new_pos = (0, 0)   # remember, center of coordinates is top left
    if direction == 'R':
        new_pos = (x+1, y) 
    elif direction == 'L':
        new_pos = (x-1, y)
    elif direction == 'U':
        new_pos = (x, y-1)
    elif direction == 'D':
        new_pos = (x, y+1)
    return new_pos 


class Mouse:
    
    def __init__(self, chromosome):
        self.chromosome = chromosome        # list of directions ('D', 'U', 'R', 'L')
        self.fitness = self.fitness_eval()  # fitness score of the chromosome
  
    @classmethod   
    def mutation(cls):  
        """randomly returns a new direction for mutation"""
        gene = random.choice(Gen_Mat)  # randomly chooses one direction
        return gene 
    
    @classmethod
    def create_genome(cls, genome_len):  # method to create a random chromosome (path)
        """create a random genome of specified length"""
        return [cls.mutation() for _ in range(genome_len)]
    
    @staticmethod
    def dynamic_mutation_rate(generation, stagnation_counter, pos):
        """Increase mutation rate if stagnation occurs and based on proximity to the goal"""
        distance_to_goal = np.abs(pos[0] - GOAL[0]) + np.abs(pos[1] - GOAL[1])
        
        # Increase mutation rate when stagnation occurs or the mouse is close to the goal
        if stagnation_counter > stagnation_mutation_change or distance_to_goal < 10:  
            return min(Mutation_rate + 0.5, 4)  # Increase mutation rate near goal or during stagnation
        return Mutation_rate
        
    def mate(self, parent2, chromosome_length):
        """mate two parents to produce a child chromosome"""
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome[:chromosome_length], parent2.chromosome[:chromosome_length]):
            prob = random.random()   # random number between 0.0 and 1.0
            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.mutation())
        return Mouse(child_chromosome)


    def fitness_eval(self):
        pos = START
        visited_positions = set()  # track visited positions to avoid loops
        valid_steps = 0
        penalty = 0  # introduce penalty for invalid moves
        last_direction = None  # Track the last direction

        reverse_moves = {'R': 'L', 'L': 'R', 'U': 'D', 'D': 'U'}
        distance_to_goal = np.abs(START[0] - GOAL[0]) + np.abs(START[1] - GOAL[1])  

        for direction in self.chromosome:
            # penalize if the mouse moves in the exact opposite direction of its last move (backtracking)
            if last_direction and reverse_moves[last_direction] == direction:
                penalty += (penalty_value * (1 + (distance_to_goal - valid_steps) / distance_to_goal)) # higher penalty if further from goal
                break

            new_pos = move(pos, direction)
            # check if the new position is a valid move in the maze
            if possible_move(maze, new_pos):
                if new_pos not in visited_positions:
                    pos = new_pos
                    visited_positions.add(pos)
                    valid_steps += 1
                    last_direction = direction
                else:
                    penalty += (penalty_value * (1 + (distance_to_goal - valid_steps) / distance_to_goal))
                    break
            else:
                break   # stop evaluation after invalid move
        
        # Manhattan distance to goal
        distance_to_goal = np.abs(pos[0] - GOAL[0]) + np.abs(pos[1] - GOAL[1])
        
        # reward progress made towards the goal (closer to the goal means lower fitness)
        progress_reward = (len(maze) + len(maze[0])) - distance_to_goal
        
        fitness = distance_to_goal * pond_dist_goal    # the lower the better
        fitness -= valid_steps * pond_valid_steps  # reward for valid steps
        fitness -= progress_reward 
        fitness += penalty

        return fitness


# Load the maze from file
maze = load_maze_from_file("maze.txt")


def main():
    print("Generic Algorithm for solving a maze\nMaze structure:\n\n")
    print_initial_maze(maze)

    generation = 1
    solution_found = False
    stagnation_counter = 0
    last_best_fitness = float('inf')
    chromosome_length = Initial_chromosome_length

    # initialize population with random chromosomes
    population = [Mouse(Mouse.create_genome(chromosome_length)) for _ in range(POPUL_Size)]
    
    while not solution_found and generation <= Generations:
        # sort by fitness (lower fitness is better)
        population.sort(key=lambda x: x.fitness)
        
        # print best solution every x generations
        if generation % 100 == 0:
            print(f"Generation: {generation}\tBest Solution: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")
        
        best_mouse = population[0]
        pos = START
        path = [pos]  # list for keeping the path taken
        visited_positions = set()  # track visited positions to avoid loops
        for move_direction in best_mouse.chromosome:
            new_pos = move(pos, move_direction)

            if possible_move(maze, new_pos) and new_pos not in visited_positions:
                pos = new_pos
                path.append(pos)
                visited_positions.add(pos)

        # check if the goal is reached
        if target_reached(pos, GOAL):
            solution_found = True
            break
        
        # check for stagnation (if no improvement in fitness)
        curent_fitness = population[0].fitness
        
        if curent_fitness < last_best_fitness:
            last_best_fitness = curent_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Increase chromosome length every x generations or after improvement
        if generation % generations_to_increase_chromosome == 0:
            chromosome_length += 1

            print(f"Increasing chromosome length to {chromosome_length}")
            for individual in population:
                extra_genes = Mouse.create_genome(1)
                individual.chromosome.extend(extra_genes)
        
        # Create a new generation
        
        new_generation = population[:POPUL_Size // 10]   # top 10% of the population
        
        # introduce new random mice to maintain diversity
        new_generation.extend([Mouse(Mouse.create_genome(chromosome_length)) for _ in range(POPUL_Size // 10)])

        remaining_population = POPUL_Size - len(new_generation)

        for _ in range(remaining_population):
            parent1 = random.choice(population[:POPUL_Size // 2])
            parent2 = random.choice(population[:POPUL_Size // 2])
            child = parent1.mate(parent2, chromosome_length)
            new_generation.append(child)

        # update mutation rate based on stagnation
        Mutation_rate = Mouse.dynamic_mutation_rate(generation, stagnation_counter, pos)
       
        population = new_generation
        generation += 1

        # cut the algorithm if it's augmenting the chromosome length too much, and restart the execution manually
        if chromosome_length == 200:
            print("No good solution")
            break

    if solution_found:
        print(f"\nSolution found in Generation {generation}\n")
        print(f"Gen: {generation}\tSolution: {''.join(population[0].chromosome)}\tFitness Score: {population[0].fitness}")
        print_maze(maze, path)
    else:
        print("\nSolution not found within the generation limit.")

main()

