from collections import defaultdict

import numpy as np
import os
import multiprocessing
import re
import copy
import random
import sys
import time

# status
INFTY = 0x3f3f3f3f
NINFTY = -0x3f3f3f3f
ALPHA = 0.2
POPULATION_SIZE = 50
init_population = 200
CPU = 1
NAME = ""
VERTICES = None
DEPOT = None
REQUIRED_EDGES = None
NON_REQUIRED_EDGES = None
VEHICLES = None
CAPACITY = None
TOTAL_COST_OF_REQUIRED_EDGES = None
RATIO = None
EDGES = None
REDGE = None
GRAPH = set()
EC = None
ED = None
SHORTEST_DIS = None


class Solution:
    def __init__(self, routes, loads, costs, total_cost, capacity):
        self.routes = routes
        self.loads = loads
        self.costs = costs
        self.total_cost = int(total_cost) if total_cost != np.inf else np.inf

        self.capacity = capacity
        self.load_exceed = sum([c - capacity for c in loads if c > capacity])

        self.is_valid = self.load_exceed == 0
        if self.is_valid:
            self.non_valid_generations = 0
        else:
            self.non_valid_generations = 1

        if self.loads:
            self.discard_prop = 2 * self.load_exceed / sum(self.loads) * pow(3, self.non_valid_generations)

    def check_valid(self):
        self.load_exceed = sum([c - self.capacity for c in self.loads if c > self.capacity])
        self.is_valid = self.load_exceed == 0
        if not self.is_valid:
            self.non_valid_generations += 1

        self.discard_prop = 2 * self.load_exceed / sum(self.loads) * pow(3, self.non_valid_generations)

        if self.routes.count([]):
            for i, c in enumerate(self.routes):
                if not c:
                    del self.routes[i]
                    del self.loads[i]
                    del self.costs[i]

    def __hash__(self):
        return hash(str(self.routes))

    def __eq__(self, other):
        return self.routes == other.routes


# -------------------------------------------------------
# Read file
# -------------------------------------------------------
def set_opt(file_name):
    global VERTICES, CAPACITY, DEPOT, REQUIRED_EDGES, NON_REQUIRED_EDGES, CPU, VEHICLES, TOTAL_COST_OF_REQUIRED_EDGES
    CPU = multiprocessing.cpu_count()
    if CPU not in range(1, 20):
        CPU = 1
    if CPU > 8:
        CPU = 8
    with open(file_name) as f:
        array = f.readlines()
        VERTICES = int(re.findall(": (.+?)\n", array[1])[0])
        DEPOT = int(re.findall(": (.+?)\n", array[2])[0])
        REQUIRED_EDGES = int(re.findall(": (.+?)\n", array[3])[0])
        NON_REQUIRED_EDGES = int(re.findall(": (.+?)\n", array[4])[0])
        VEHICLES = int(re.findall(": (.+?)\n", array[5])[0])
        CAPACITY = int(re.findall(": (.+?)\n", array[6])[0])
        TOTAL_COST_OF_REQUIRED_EDGES = int(re.findall(": (.+?)\n", array[7])[0])
        global EDGES, EC, ED, RATIO, SHORTEST_DIS, SHORTEST_PATH, REDGE
        SHORTEST_DIS = np.full((VERTICES + 1, VERTICES + 1), fill_value=INFTY, dtype=int)
        # SHORTEST_PATH = np.full((VERTICES + 1, VERTICES + 1), fill_value=None, dtype=list)
        # np.fill_diagonal(SHORTEST_PATH, [])
        EDGES = defaultdict(list)
        EC = {}
        ED = {}
        REDGE = []
        RATIO = 0
        for line in array[9:-1]:
            line = line.strip().split()
            head = int(line[0])
            tail = int(line[1])
            GRAPH.add(head)
            GRAPH.add(tail)
            cost = int(line[2])
            demand = int(line[3])
            EDGES[head].append(tail)
            EDGES[tail].append(head)
            EC[(tail, head)] = cost
            EC[(head, tail)] = cost
            SHORTEST_DIS[head][tail] = cost
            SHORTEST_DIS[tail][head] = cost
            # SHORTEST_PATH[head][tail] = []
            # SHORTEST_PATH[tail][head] = []
            # SHORTEST_PATH[head][tail].append((head, tail))
            # SHORTEST_PATH[tail][head].append((tail, head))
            ED[(head, tail)] = demand
            ED[(tail, head)] = demand
            if demand:
                REDGE.append((head, tail))
                REDGE.append((tail, head))
            RATIO += demand
        RATIO = VEHICLES / RATIO


def init_path():
    global SHORTEST_DIS
    gra = list(GRAPH)
    for k in gra:
        for i in gra:
            for j in gra:
                tmp = SHORTEST_DIS[i][k] + SHORTEST_DIS[k][j]
                if SHORTEST_DIS[i][j] > tmp:
                    SHORTEST_DIS[i][j] = tmp

                    # SHORTEST_PATH[i][j] = SHORTEST_PATH[i][k] + SHORTEST_PATH[k][j]
                # -------------------------------------------------------
    np.fill_diagonal(SHORTEST_DIS, 0)


# Population initialization
# -------------------------------------------------------
def population_init(REDGE, ED, EC, SHORTEST_DIS, num, CAPACITY, DEPOT, seed):
    population = []
    # random.seed(seed)
    while len(population) < num:
        sol = RPSH(REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)
        if random.random() > sol.discard_prop:
            population.append(sol)
    return population


# -------------------------------------------------------
# Random Path Scanning Heuristic
# -------------------------------------------------------
def RPSH(REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT):
    paths = []
    rearc = REDGE.copy()
    random.shuffle(rearc)  # Required arc, free
    vehicles = -1  # vehicle number
    loads = []  # loads of every vehicle
    costs = []  # costs of every path
    while True:
        src = DEPOT
        vehicles += 1
        load = 0
        cost = 0
        path = []
        while True:
            cost_add = INFTY
            edge_add = False
            for edge in rearc:
                if load + ED[edge] <= CAPACITY:
                    d_se = SHORTEST_DIS[src][edge[0]]
                    if d_se < cost_add:
                        cost_add = d_se
                        edge_add = edge
                    elif d_se == cost_add and better(edge, edge_add, DEPOT, load, ED, EC, SHORTEST_DIS,
                                                     CAPACITY):
                        edge_add = edge
            if edge_add:
                # path.extend(SHORTEST_PATH[(src,edge_add[0])])
                path.append(edge_add)
                rearc.remove(edge_add)
                rearc.remove(inverseArc(edge_add))
                load += ED[edge_add]
                cost += (EC[edge_add] + cost_add)
                src = edge_add[1]
            if len(rearc) == 0 or cost_add == INFTY:
                break
        cost += SHORTEST_DIS[src][DEPOT]
        costs.append(cost)
        loads.append(load)
        # path.extend(SHORTEST_PATH[(src, DEPOT)])
        paths.append(path)
        if len(rearc) == 0:
            break
    solution = Solution(paths, loads, costs, sum(costs), CAPACITY)
    return solution


def better(tmp_edge, edge, src, load, ED, EC, SHORTEST_DIS, CAPACITY, rule=None):
    if rule is None:
        rule = random.randint(1, 12)
    if not edge:
        return True
    else:
        if rule < 3:
            return check_ratio(tmp_edge, edge, ED, EC,
                               isMax=True)
        elif rule < 5:
            return check_ratio(tmp_edge, edge, ED, EC, isMax=False)
        else:
            tmp_edge_cost = SHORTEST_DIS[tmp_edge[1]][src]
            edge_cost = SHORTEST_DIS[edge[1]][src]
            if rule < 7:  # maximize return cost
                return tmp_edge_cost > edge_cost
            elif rule < 9:  # minimize return cost
                return tmp_edge_cost < edge_cost
            elif rule < 11:
                if load > CAPACITY / 2:
                    return tmp_edge_cost > edge_cost
                else:  # else apply rule 4
                    return tmp_edge_cost < edge_cost
            else:
                return random.randint(0, 1)


# -------------------------------------------------------
# get inverse arc
# -------------------------------------------------------
def inverseArc(arc):
    """
    :paraam arc: edge
    """
    return arc[::-1]


def check_ratio(tmp_edge, edge, ED, EC, isMax=False):
    tmp_edge_ratio = EC[tmp_edge] / ED[tmp_edge]
    edge_ratio = EC[edge] / ED[edge]
    if isMax:
        return tmp_edge_ratio > edge_ratio  # if tmp_edge has larger ratio, then return True else False
    else:
        return tmp_edge_ratio < edge_ratio  # if tmp_edge has smaller ratio, then return True else False


def cal_total_cost(entities):
    cost = 0
    for i in entities.keys():
        cost += cal_cost(entities[i])
    return cost


# -------------------------------------------------------
# Calculate the demand for a given path
# -------------------------------------------------------
def cal_demand(path):
    demand = 0
    for edge in path:
        demand += ED[edge]
    return demand


# -------------------------------------------------------
# Calculate the cost for a given path
# -------------------------------------------------------
def cal_cost(paths):
    costs = 0
    for path in paths:
        for edge in path:
            costs += EC[edge]
        for edge_index in range(len(path) - 1):
            costs += SHORTEST_DIS[path[edge_index][1]][path[edge_index + 1][0]]
        costs += SHORTEST_DIS[DEPOT][path[0][0]] + SHORTEST_DIS[path[-1][1]][DEPOT]
    return costs


def single_insertion(solution):
    # get selected task index
    new_solution = copy.deepcopy(solution)
    routes: list = new_solution.routes
    selected_arc_index = random.randrange(0, len(routes))  # start <= N < end
    selected_arc = routes[selected_arc_index]

    selected_task_index = random.randrange(0, len(selected_arc))  # start <= N < end

    # information used in calculation
    u, v = selected_arc[selected_task_index]
    task = (u, v)

    # calculate changed selected arc costs
    pre_end = selected_arc[selected_task_index - 1][1] if selected_task_index != 0 else DEPOT
    next_start = selected_arc[selected_task_index + 1][0] if selected_task_index != len(
        selected_arc) - 1 else DEPOT

    changed_cost = SHORTEST_DIS[pre_end][next_start] - SHORTEST_DIS[pre_end][u] - SHORTEST_DIS[
        v][next_start] - EC[task]

    new_solution.costs[selected_arc_index] += changed_cost
    new_solution.total_cost += changed_cost
    new_solution.loads[selected_arc_index] -= ED[task]

    selected_task = selected_arc.pop(selected_task_index)

    # get inserted index
    routes.append([])
    inserting_arc_index = random.randrange(0, len(routes))
    inserting_arc = routes[inserting_arc_index]
    inserting_position = random.randint(0, len(inserting_arc))  # start <= N <= end

    # calculate changed inserted arc costs
    pre_end = inserting_arc[inserting_position - 1][1] if inserting_position != 0 else DEPOT
    next_start = inserting_arc[inserting_position][0] if inserting_position != len(inserting_arc) else DEPOT

    changed_cost = SHORTEST_DIS[pre_end][u] + SHORTEST_DIS[v][next_start] + EC[task] - SHORTEST_DIS[
        pre_end][next_start]
    reversed_changed_cost = SHORTEST_DIS[pre_end][v] + SHORTEST_DIS[u][next_start] + EC[task] - SHORTEST_DIS[
        pre_end][next_start]  # (v, u)
    if reversed_changed_cost < changed_cost:
        selected_task = (v, u)
        changed_cost = reversed_changed_cost

    if not inserting_arc:  # means a new arc
        new_solution.costs.append(changed_cost)
        new_solution.loads.append(ED[task])
    else:
        del routes[-1]
        new_solution.costs[inserting_arc_index] += changed_cost
        new_solution.loads[inserting_arc_index] += ED[task]
    new_solution.total_cost += changed_cost

    inserting_arc.insert(inserting_position, selected_task)

    new_solution.check_valid()

    return new_solution


def double_insertion(solution):
    # get selected first task index
    new_solution = copy.deepcopy(solution)
    routes: list = new_solution.routes
    selected_arc_index = random.randrange(0, len(routes))  # start <= N < end
    while len(routes[selected_arc_index]) < 2:  # routes that size >= 2 can be applied DI
        selected_arc_index = random.randrange(0, len(routes))

    selected_arc = routes[selected_arc_index]
    selected_task_index = random.randrange(0, len(
        selected_arc) - 1)  # start <= N < end - 1, should leave a position for second

    # information used in calculation
    u1, v1 = selected_arc[selected_task_index]
    u2, v2 = selected_arc[selected_task_index + 1]
    task1 = (u1, v1)
    task2 = (u2, v2)

    # calculate changed selected arc costs
    pre_end = selected_arc[selected_task_index - 1][1] if selected_task_index != 0 else DEPOT
    next_start = selected_arc[selected_task_index + 2][0] if selected_task_index != len(
        selected_arc) - 2 else DEPOT

    changed_cost = SHORTEST_DIS[pre_end][next_start] \
                   - SHORTEST_DIS[pre_end][u1] - EC[task1] - SHORTEST_DIS[v1][u2] - EC[task2] - SHORTEST_DIS[
                       v2][next_start]
    new_solution.costs[selected_arc_index] += changed_cost
    new_solution.total_cost += changed_cost
    new_solution.loads[selected_arc_index] -= ED[task1] + ED[task2]

    selected_task1 = selected_arc.pop(selected_task_index)
    selected_task2 = selected_arc.pop(selected_task_index)

    # get inserted index
    routes.append([])
    inserting_arc_index = random.randrange(0, len(routes))
    inserting_arc = routes[inserting_arc_index]
    inserting_position = random.randint(0, len(inserting_arc))  # start <= N <= end

    # calculate changed inserted arc costs
    pre_end = inserting_arc[inserting_position - 1][1] if inserting_position != 0 else DEPOT
    next_start = inserting_arc[inserting_position][0] if inserting_position != len(inserting_arc) else DEPOT

    changed_cost = SHORTEST_DIS[pre_end][u1] + EC[task1] + SHORTEST_DIS[v1][u2] + EC[task2] + SHORTEST_DIS[
        v2][next_start] \
                   - SHORTEST_DIS[pre_end][next_start]
    reversed_changed_cost = SHORTEST_DIS[pre_end][v2] + EC[task2] + SHORTEST_DIS[u2][v1] + EC[task1] + \
                            SHORTEST_DIS[u1][next_start] \
                            - SHORTEST_DIS[pre_end][next_start]
    if reversed_changed_cost < changed_cost:
        selected_task1 = (v2, u2)
        selected_task2 = (v1, u1)
        changed_cost = reversed_changed_cost

    if not inserting_arc:  # means a new arc
        new_solution.costs.append(changed_cost)
        new_solution.loads.append(ED[task1] + ED[task2])
    else:
        del routes[-1]
        new_solution.costs[inserting_arc_index] += changed_cost
        new_solution.loads[inserting_arc_index] += ED[task2] + ED[task1]
    new_solution.total_cost += changed_cost

    inserting_arc.insert(inserting_position, selected_task2)
    inserting_arc.insert(inserting_position, selected_task1)
    if sum(solution.costs)== solution.total_cost:
        print(solution.is_valid)
    new_solution.check_valid()

    return new_solution


def swap(solution):
    new_solution = copy.deepcopy(solution)
    routes: list = new_solution.routes

    # get first selected task index
    selected_arc_index1 = random.randrange(0, len(routes))  # start <= N < end
    selected_arc1 = routes[selected_arc_index1]
    selected_task_index1 = random.randrange(0, len(selected_arc1))  # start <= N < end

    # get second selected task index
    selected_arc_index2 = random.randrange(0, len(routes))  # start <= N < end
    selected_arc2 = routes[selected_arc_index2]
    selected_task_index2 = random.randrange(0, len(selected_arc2))  # start <= N < end
    while selected_arc_index1 == selected_arc_index2 and selected_task_index1 == selected_task_index2:
        selected_arc_index2 = random.randrange(0, len(routes))  # start <= N < end
        selected_arc2 = routes[selected_arc_index2]
        selected_task_index2 = random.randrange(0, len(selected_arc2))  # start <= N < end

    # information used in calculation
    u1, v1 = selected_arc1[selected_task_index1]
    u2, v2 = selected_arc2[selected_task_index2]
    task1 = (u1, v1)
    task2 = (u2, v2)

    pre_end1 = selected_arc1[selected_task_index1 - 1][1] if selected_task_index1 != 0 else DEPOT
    next_start1 = selected_arc1[selected_task_index1 + 1][0] if selected_task_index1 != len(
        selected_arc1) - 1 else DEPOT
    pre_end2 = selected_arc2[selected_task_index2 - 1][1] if selected_task_index2 != 0 else DEPOT
    next_start2 = selected_arc2[selected_task_index2 + 1][0] if selected_task_index2 != len(
        selected_arc2) - 1 else DEPOT

    selected_task1 = selected_arc1.pop(selected_task_index1)
    if selected_arc_index1 == selected_arc_index2 and selected_task_index1 < selected_task_index2:
        selected_task2 = selected_arc2.pop(selected_task_index2 - 1)
    else:
        selected_task2 = selected_arc2.pop(selected_task_index2)

    # first arc cost change : insert task2 into arc1
    reduced_cost1 = SHORTEST_DIS[pre_end1][u1] + EC[task1] + SHORTEST_DIS[v1][next_start1]
    changed_cost1 = SHORTEST_DIS[pre_end1][u2] + EC[task2] + SHORTEST_DIS[v2][next_start1] - reduced_cost1
    reversed_changed_cost1 = SHORTEST_DIS[pre_end1][v2] + EC[task2] + SHORTEST_DIS[u2][next_start1] - reduced_cost1
    if reversed_changed_cost1 < changed_cost1:
        selected_task2 = (v2, u2)
        changed_cost1 = reversed_changed_cost1

    new_solution.costs[selected_arc_index1] += changed_cost1
    new_solution.total_cost += changed_cost1
    new_solution.loads[selected_arc_index1] += ED[task2] - ED[task1]

    selected_arc1.insert(selected_task_index1, selected_task2)

    # second arc cost change : insert task1 into arc2
    reduced_cost2 = SHORTEST_DIS[pre_end2][u2] + EC[task2] + SHORTEST_DIS[v2][next_start2]
    changed_cost2 = SHORTEST_DIS[pre_end2][u1] + EC[task1] + SHORTEST_DIS[v1][next_start2] - reduced_cost2
    reversed_changed_cost2 = SHORTEST_DIS[pre_end2][v1] + EC[task1] + SHORTEST_DIS[u1][next_start2] - reduced_cost2
    if reversed_changed_cost2 < changed_cost2:
        selected_task1 = (v1, u1)
        changed_cost2 = reversed_changed_cost2

    new_solution.costs[selected_arc_index2] += changed_cost2
    new_solution.total_cost += changed_cost2
    new_solution.loads[selected_arc_index2] += ED[task1] - ED[task2]

    selected_arc2.insert(selected_task_index2, selected_task1)

    if selected_arc_index1 == selected_arc_index2:
        new_solution.total_cost = cal_cost(new_solution.routes)

    new_solution.check_valid()

    return new_solution


def searching(population):
    # a=time.time()
    popu = population.copy()
    for individual in popu:
        if random.random() > individual.discard_prop:
            if random.random() > ALPHA:
                population.add(local_search(individual))
        else:
            population.remove(individual)
    while len(population) > POPULATION_SIZE:
        worst_individual = max(population, key=lambda x: x.total_cost)
        population.remove(worst_individual)
    valid_population = [p for p in population if p.is_valid]
    best_individual = min(valid_population, key=lambda x: x.total_cost)
    # print(time.time()-a)
    return best_individual


def local_search(solution):
    new_solution = False
    while not new_solution:
        new_solution = min([move(solution) for move in [single_insertion, double_insertion, swap]],
                           key=lambda x: x.total_cost)
        discard_prop = 0 if new_solution.is_valid else 0.6
        if random.random() < discard_prop:
            new_solution = False

    return new_solution





def command_line(argv):
    file_name = argv[0]
    termination = int(argv[2])
    seed = int(argv[4])
    return file_name, termination, seed





def ini_ps():
    pool = multiprocessing.Pool()
    result = []
    num = int(init_population)
    for i in range(CPU):
        result.append((pool.apply_async(population_init, args=(
            REDGE, ED, EC, SHORTEST_DIS, num, CAPACITY, DEPOT, SEED + i,))))
    pool.close()
    pool.join()
    sp = []
    for x in result:
        sp.extend(x.get())
    return set(sp)


if __name__ == "__main__":
    start = time.time()
    global SEED
    file_name, termination, SEED = command_line(sys.argv[1:])
    random.seed(SEED)
    set_opt(file_name)
    init_path()
    pop = ini_ps()
    end = start + termination - 1
    # while len(u)<=POPULATION_SIZE:
    #     for p in pop:
    #         new_solution = random.choice([single_insertion, double_insertion, swap])(p)
    #         if random.random() > new_solution.discard_prop:
    #             u.add(new_solution)
    while True:
        best = searching(pop)
        print(best.total_cost)
        if time.time() > end:
            break
    result = ""
    routes = []
    for i in best.routes:
        routes += [0] + i + [0]
    for item in routes:
        result += str(item) + ","
    result = result.replace(' ', '')
    print("s", result[:-1])
    print("q " + str(best.total_cost))
    un_time = (time.time() - start)
