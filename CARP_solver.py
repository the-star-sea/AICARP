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
ALPHA = 0.2  # 0.2
POPULATION_SIZE = 30  # 30
init_population = 500  # 500
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
TIME = 60  # 60


class Solution:
    def __init__(self, routes, loads, costs, total_cost, capacity):
        self.routes = routes
        self.loads = loads
        # self.costs = costs
        self.total_cost = int(total_cost) if total_cost != np.inf else np.inf
        self.capacity = capacity
        self.load_exceed = sum([c - capacity for c in loads if c > capacity])

        self.is_valid = self.load_exceed == 0
        if self.is_valid:
            self.nvg = 0
        else:
            self.nvg = 1

        if self.loads:
            self.discard_prop = 2 * self.load_exceed / sum(self.loads) * pow(3, self.nvg)

    def init_valid(self):
        self.load_exceed = sum([c - self.capacity for c in self.loads if c > self.capacity])
        self.is_valid = self.load_exceed == 0
        if not self.is_valid:
            self.nvg += 1

        self.discard_prop = 2 * self.load_exceed / sum(self.loads) * pow(3, self.nvg)

        if self.routes.count([]):
            for i, c in enumerate(self.routes):
                if not c:
                    del self.routes[i]
                    del self.loads[i]
                    # del self.costs[i]

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
        global POPULATION_SIZE
        # if VERTICES < 50:
        #     POPULATION_SIZE = 50
        # else:
        #     POPULATION_SIZE=30
        global EDGES, EC, ED, RATIO, SHORTEST_DIS, SHORTEST_PATH, REDGE
        SHORTEST_DIS = np.full((VERTICES + 1, VERTICES + 1), fill_value=INFTY, dtype=int)
        # SHORTEST_PATH = np.full((VERTICES + 1, VERTICES + 1), fill_value=None, dtype=list)
        # np.fill_diagonal(SHORTEST_PATH, [])
        EDGES = defaultdict(list)
        EC = {}
        ED = {}
        REDGE = []
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
def population_init(REDGE, ED, EC, SHORTEST_DIS, num, CAPACITY, DEPOT, seed, end, POPULATION_SIZE, VEHICLE):
    population = set()
    # random.seed(seed)
    while len(population) < num:
        sol = RPSH(REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)
        if random.random() > sol.discard_prop:
            population.add(sol)
    best = searching(population, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, seed, POPULATION_SIZE, end, VEHICLE)
    return best


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
def cal_cost(paths, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT):
    costs = 0
    for path in paths:

        for edge in path:
            costs += EC[edge]

        for edge_index in range(len(path) - 1):
            costs += SHORTEST_DIS[path[edge_index][1]][path[edge_index + 1][0]]

        costs += SHORTEST_DIS[DEPOT][path[0][0]] + SHORTEST_DIS[path[-1][1]][DEPOT]

    return costs


# def cal_cost1(paths, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT):
#     costs = 0
#     a = []
#     for path in paths:
#         u = 0
#         for edge in path:
#             costs += EC[edge]
#             u += EC[edge]
#         for edge_index in range(len(path) - 1):
#             costs += SHORTEST_DIS[path[edge_index][1]][path[edge_index + 1][0]]
#             u += SHORTEST_DIS[path[edge_index][1]][path[edge_index + 1][0]]
#         costs += SHORTEST_DIS[DEPOT][path[0][0]] + SHORTEST_DIS[path[-1][1]][DEPOT]
#         u += SHORTEST_DIS[DEPOT][path[0][0]] + SHORTEST_DIS[path[-1][1]][DEPOT]
#         a.append(u)
#     return costs, a


def single_insertion(solution, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE):
    # get selected task index
    new_solution = copy.deepcopy(solution)
    routes: list = new_solution.routes
    sai = random.randrange(0, len(routes))  # start <= N < end
    sa = routes[sai]

    sti = random.randrange(0, len(sa))  # start <= N < end

    # information used in calculation
    u, v = sa[sti]
    task = (u, v)

    # calculate changed selected arc costs
    pre_end = sa[sti - 1][1] if sti != 0 else DEPOT
    next_start = sa[sti + 1][0] if sti != len(
        sa) - 1 else DEPOT

    cc = SHORTEST_DIS[pre_end][next_start] - SHORTEST_DIS[pre_end][u] - SHORTEST_DIS[
        v][next_start] - EC[task]

    # new_solution.costs[sai] += cc
    new_solution.total_cost += cc
    new_solution.loads[sai] -= ED[task]

    selected_task = sa.pop(sti)

    # get inserted index
    routes.append([])
    iai = random.randrange(0, len(routes))
    ia = routes[iai]
    ip = random.randint(0, len(ia))  # start <= N <= end

    # calculate changed inserted arc costs
    pre_end = ia[ip - 1][1] if ip != 0 else DEPOT
    next_start = ia[ip][0] if ip != len(ia) else DEPOT

    cc = SHORTEST_DIS[pre_end][u] + SHORTEST_DIS[v][next_start] + EC[task] - SHORTEST_DIS[
        pre_end][next_start]
    rcc = SHORTEST_DIS[pre_end][v] + SHORTEST_DIS[u][next_start] + EC[task] - SHORTEST_DIS[
        pre_end][next_start]  # (v, u)
    if rcc < cc:
        selected_task = (v, u)
        cc = rcc

    if not ia:  # means a new arc
        # new_solution.costs.append(cc)
        new_solution.loads.append(ED[task])
    else:
        del routes[-1]
        # new_solution.costs[iai] += cc
        new_solution.loads[iai] += ED[task]
    new_solution.total_cost += cc

    ia.insert(ip, selected_task)

    new_solution.init_valid()
    # if sum(solution.costs)== solution.total_cost:
    #     print(solution.is_valid)
    # if cal_cost(solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)!=sum(solution.costs):
    #     print("sb")
    # if cal_cost(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)!=sum(new_solution.costs):
    #     print("sb11111111111111111111111111")
    return new_solution


def double_insertion(solution, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE):
    # get selected first task index
    new_solution = copy.deepcopy(solution)
    routes: list = new_solution.routes
    sai = random.randrange(0, len(routes))  # start <= N < end
    while len(routes[sai]) < 2:  # routes that size >= 2 can be applied DI
        sai = random.randrange(0, len(routes))

    sa = routes[sai]
    sti = random.randrange(0, len(
        sa) - 1)  # start <= N < end - 1, should leave a position for second

    # information used in calculation
    u1, v1 = sa[sti]
    u2, v2 = sa[sti + 1]
    task1 = (u1, v1)
    task2 = (u2, v2)

    # calculate changed selected arc costs
    pre_end = sa[sti - 1][1] if sti != 0 else DEPOT
    next_start = sa[sti + 2][0] if sti != len(
        sa) - 2 else DEPOT

    cc = SHORTEST_DIS[pre_end][next_start] \
         - SHORTEST_DIS[pre_end][u1] - EC[task1] - SHORTEST_DIS[v1][u2] - EC[task2] - SHORTEST_DIS[
             v2][next_start]
    # new_solution.costs[sai] += cc
    new_solution.total_cost += cc
    new_solution.loads[sai] -= ED[task1] + ED[task2]

    t1 = sa.pop(sti)
    t2 = sa.pop(sti)

    routes.append([])
    iai = random.randrange(0, len(routes))
    ia = routes[iai]
    ip = random.randint(0, len(ia))  # start <= N <= end

    pre_end = ia[ip - 1][1] if ip != 0 else DEPOT
    next_start = ia[ip][0] if ip != len(ia) else DEPOT

    cc = SHORTEST_DIS[pre_end][u1] + EC[task1] + SHORTEST_DIS[v1][u2] + EC[task2] + SHORTEST_DIS[
        v2][next_start] \
         - SHORTEST_DIS[pre_end][next_start]
    rcc = SHORTEST_DIS[pre_end][v2] + EC[task2] + SHORTEST_DIS[u2][v1] + EC[task1] + \
          SHORTEST_DIS[u1][next_start] \
          - SHORTEST_DIS[pre_end][next_start]
    if rcc < cc:
        t1 = (v2, u2)
        t2 = (v1, u1)
        cc = rcc

    if not ia:  # means a new arc
        # new_solution.costs.append(cc)
        new_solution.loads.append(ED[task1] + ED[task2])
    else:
        del routes[-1]
        # new_solution.costs[iai] += cc
        new_solution.loads[iai] += ED[task2] + ED[task1]
    new_solution.total_cost += cc

    ia.insert(ip, t2)
    ia.insert(ip, t1)

    new_solution.init_valid()
    # if sum(solution.costs)== solution.total_cost:
    #     print(solution.is_valid)
    # if cal_cost(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)!=sum(new_solution.costs):
    #     print("sb11111111111111111111111111")
    return new_solution


def swap(solution, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE):
    new_solution = copy.deepcopy(solution)
    routes: list = new_solution.routes
    # a, _ = cal_cost1(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)
    # get first selected task index
    sai1 = random.randrange(0, len(routes))  # start <= N < end
    sa1 = routes[sai1]
    sti1 = random.randrange(0, len(sa1))  # start <= N < end

    # get second selected task index
    sai2 = random.randrange(0, len(routes))  # start <= N < end
    sa2 = routes[sai2]
    sti2 = random.randrange(0, len(sa2))  # start <= N < end
    while sai1 == sai2 and sti1 == sti2:
        sai2 = random.randrange(0, len(routes))  # start <= N < end
        sa2 = routes[sai2]
        sti2 = random.randrange(0, len(sa2))  # start <= N < end

    # information used in calculation
    u1, v1 = sa1[sti1]
    u2, v2 = sa2[sti2]
    task1 = (u1, v1)
    task2 = (u2, v2)

    pre_end1 = sa1[sti1 - 1][1] if sti1 != 0 else DEPOT
    next_start1 = sa1[sti1 + 1][0] if sti1 != len(
        sa1) - 1 else DEPOT
    pre_end2 = sa2[sti2 - 1][1] if sti2 != 0 else DEPOT
    next_start2 = sa2[sti2 + 1][0] if sti2 != len(
        sa2) - 1 else DEPOT

    selected_task1 = sa1.pop(sti1)
    if sai1 == sai2 and sti1 < sti2:
        selected_task2 = sa2.pop(sti2 - 1)
    else:
        selected_task2 = sa2.pop(sti2)

    # first arc cost change : insert task2 into arc1
    rc1 = SHORTEST_DIS[pre_end1][u1] + EC[task1] + SHORTEST_DIS[v1][next_start1]
    cc1 = SHORTEST_DIS[pre_end1][u2] + EC[task2] + SHORTEST_DIS[v2][next_start1] - rc1
    rcc1 = SHORTEST_DIS[pre_end1][v2] + EC[task2] + SHORTEST_DIS[u2][next_start1] - rc1
    if rcc1 < cc1:
        selected_task2 = (v2, u2)
        cc1 = rcc1

    # new_solution.costs[sai1] += cc1
    new_solution.total_cost += cc1
    new_solution.loads[sai1] += ED[task2] - ED[task1]

    sa1.insert(sti1, selected_task2)

    # second arc cost change : insert task1 into arc2
    rc2 = SHORTEST_DIS[pre_end2][u2] + EC[task2] + SHORTEST_DIS[v2][next_start2]
    cc2 = SHORTEST_DIS[pre_end2][u1] + EC[task1] + SHORTEST_DIS[v1][next_start2] - rc2
    rcc2 = SHORTEST_DIS[pre_end2][v1] + EC[task1] + SHORTEST_DIS[u1][next_start2] - rc2
    if rcc2 < cc2:
        selected_task1 = (v1, u1)
        cc2 = rcc2

    # new_solution.costs[sai2] += cc2
    new_solution.total_cost += cc2
    new_solution.loads[sai2] += ED[task1] - ED[task2]

    sa2.insert(sti2, selected_task1)

    if sai1 == sai2:
        new_solution.total_cost = cal_cost(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)
    # b, _ = cal_cost1(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)
    # if (sum(new_solution.costs)-sum(solution.costs))!=(new_solution.total_cost-solution.total_cost):
    #     print(sai1==sai2)
    new_solution.init_valid()
    # if sum(solution.costs)== solution.total_cost:
    #     print(solution.is_valid)
    # if cal_cost(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)!=new_solution.total_cost:
    #     print("sb")
    # if cal_cost(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)!=sum(new_solution.costs):
    #     print("sb11111111111111111111111111")
    return new_solution


def opt(solution: Solution, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE):
    new_solution = copy.deepcopy(solution)
    routes: list = new_solution.routes
    sai = random.randrange(0, len(routes))  # start <= N < end
    sa = routes[sai]
    sti = random.randrange(0, len(sa))  # start <= N < end

    # information used in calculation
    u, v = sa[sti]
    task = (u, v)
    pre_end = sa[sti - 1][1] if sti != 0 else DEPOT
    next_start = sa[sti + 1][0] if sti != len(sa) - 1 else DEPOT
    new_solution.total_cost += (SHORTEST_DIS[pre_end][v] - SHORTEST_DIS[pre_end][u] + SHORTEST_DIS[u][next_start] -
                                SHORTEST_DIS[v][next_start])
    sa[sti] = (v, u)
    new_solution.init_valid()
    # if cal_cost(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)!=new_solution.total_cost:
    #     print("sb11111111111111111111111111")
    return new_solution


def MS(solution, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE):
    new_solution: Solution = copy.deepcopy(solution)
    routes: list = new_solution.routes
    while (True):
        # if random.random()>ALPHA:
        #     verhicles = random.randrange(1, int(len(routes)/2))
        # else:
        vehicles = random.randrange(1, int(len(routes)))
        verhicles1 = vehicles
        # if verhicles>5:
        #     if random.random() > 0.3:
        #         verhicles = int(verhicles / 3)
        if verhicles1 > 3:
            if random.random() > 0.1:
                vehicles = int(verhicles1 / 2)
        # if verhicles1>5:
        #     if random.random() > 0.2:
        #         verhicles = int(verhicles1 / 3)

        vehicle = random.sample(range(0, len(routes)), vehicles)
        arcs = []
        for i in range(vehicles):
            sai = vehicle[i]
            sa = routes[sai]
            arcs.extend(sa)
        arct = arcs.copy()
        for ob in arct:
            arcs.append(inverseArc(ob))
        tempt_solution = RPSH(arcs, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)
        # if len(tempt_solution.routes) <= vehicles+VEHICLE-len(solution.routes):
        if len(tempt_solution.routes) <= vehicles:
            break
    vehicle = sorted(vehicle, reverse=True)
    for ob in vehicle:
        del new_solution.routes[ob]
        del new_solution.loads[ob]
    for i in range(len(tempt_solution.routes)):
        new_solution.routes.append(tempt_solution.routes[i])
        new_solution.loads.append(tempt_solution.loads[i])

        # new_solution.costs[vehicle[i]] = tempt_solution.costs[i]
    new_solution.total_cost = cal_cost(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)
    # print(new_solution.costs[vehicle[0]])
    # a, u = cal_cost1(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT)
    # # uu=new_solution.costs
    # if a != new_solution.total_cost:
    #     print("sb")
    new_solution.init_valid()

    # print(cal_cost(new_solution.routes, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT) )
    return new_solution


def searching(population, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, SEED, POPULATION_SIZE, end, VEHICLE):
    # random.seed(SEED)
    while True:
        best = search(population, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, POPULATION_SIZE, VEHICLE)
        # if SEED == 1001:
        #     print(SEED, best[0].total_cost)
        if time.time() > end:
            break
    return best


def search(population, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, POPULATION_SIZE, VEHICLE):
    popu = population.copy()
    for individual in popu:
        if random.random() > individual.discard_prop:
            if random.random() > ALPHA:
                population.add(local_search(individual, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE))
        else:
            population.remove(individual)
    while len(population) > POPULATION_SIZE:
        worst_individual = max(population, key=lambda x: x.total_cost)
        population.remove(worst_individual)
    valid_population = [p for p in population if p.is_valid]
    # print(len(100*valid_population)/len(population))
    # if len(valid_population)==0:
    #     print(len(population))
    best_individual = min(valid_population, key=lambda x: x.total_cost)
    return best_individual, population


def local_search(solution: Solution, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE):
    new_solution = False
    while not new_solution:
        # [single_insertion, double_insertion, swap,opt]
        if solution.is_valid:
            new_solution = min([move(solution, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE) for move in
                                [single_insertion, double_insertion, swap, MS]],
                               key=lambda x: x.total_cost)
        else:
            new_solution = min([move(solution, REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, VEHICLE) for move in
                                [single_insertion, double_insertion, swap]],
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


#
# def extend_search(pop, end):
#     result = []
#     # end1 = time.time() + TIME
#     # while end1 < end:
#     #     pool = multiprocessing.Pool()
#     #     for i in range(CPU):
#     #         result.append(
#     #             (pool.apply_async(searching,
#     #                               args=(pop[i], REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, SEED  + i, POPULATION_SIZE,
#     #                                     end1,))))
#     #     pool.close()
#     #     pool.join()
#     #     bests = []
#     #     for i in result:
#     #         bests.extend(list(i.get()[1]))
#     #     random.shuffle(bests)
#     #     m = int(len(bests) / CPU)
#     #     pop = []
#     #     for i in range(0, len(bests), m):
#     #         pop.append(set(bests[i:i + m]))
#     #     end1 += TIME
#     pool = multiprocessing.Pool()
#     for i in range(CPU):
#         result.append(
#             (pool.apply_async(searching,
#                               args=(
#                                   pop[i], REDGE, ED, EC, SHORTEST_DIS, CAPACITY, DEPOT, SEED + i, POPULATION_SIZE,
#                                   end,))))
#     pool.close()
#     pool.join()
#     bests = []
#     for i in result:
#         bests.append(i.get()[0])
#     best_individual = min(bests, key=lambda x: x.total_cost)
#     return best_individual


def ini_ps(end):
    pool = multiprocessing.Pool()
    result = []
    num = int(init_population)
    population_size = int(POPULATION_SIZE)
    for i in range(CPU):
        result.append((pool.apply_async(population_init, args=(
            REDGE, ED, EC, SHORTEST_DIS, num, CAPACITY, DEPOT, SEED + i, end, population_size, VEHICLES,))))
    pool.close()
    pool.join()
    bests = []
    for i in result:
        bests.append(i.get()[0])
    best_individual = min(bests, key=lambda x: x.total_cost)
    # sp = []
    # for ind in result:
    #     sp.append(ind.get())
    return best_individual


if __name__ == "__main__":
    start = time.time()
    global SEED
    file_name, termination, SEED = command_line(sys.argv[1:])
    random.seed(SEED)
    set_opt(file_name)
    init_path()
    end = start + termination - 0.5
    best = ini_ps(end)

    # best = extend_search(pop, end)

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
