from z3 import *
import random
import pygame
import time
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import math
import json

NUM_ROOMS = 30
number_of_rooms = NUM_ROOMS

SCALE_FACTOR = 1000

ROOM_WIDTH_MIN = 10
ROOM_WIDTH_MAX = 20
ROOM_HEIGHT_MIN = 10 * SCALE_FACTOR
ROOM_HEIGHT_MAX = 20 * SCALE_FACTOR

CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

SEPARATION = 10
SEPARATION_Y = SEPARATION * SCALE_FACTOR

BORDER = 5
LINEWIDTH = 30
LINEWIDTH_Y = LINEWIDTH * SCALE_FACTOR

EXCEPTION_RATE = 0

GRID_CELL = 5
GRID_CELL_Y = GRID_CELL * SCALE_FACTOR
grid_counts = []
NUM_LOOPS = 5
NUM_RUNS = 25

PASSAGE_WIDTH = 3


quad_constraints = False
line_constraints = False
big_room_constraint = False
show_delaunay = False
show_sparse = False

rooms = []
assumptions = []
directions = [{"vert": "above", "horiz": "left"},
              {"vert": "above", "horiz": "same"},
              {"vert": "above", "horiz": "right"},
              {"vert": "same", "horiz": "left"},
              {"vert": "same", "horiz": "right"},
              {"vert": "below", "horiz": "left"},
              {"vert": "below", "horiz": "same"},
              {"vert": "below", "horiz": "right"}]

timing_info = {}
accumulated_timing_info = {}
timing_per_run = {}
and_clause_count = 0
or_clause_count = 0

def init_rooms():
    global rooms
    rooms = []
    for i in range(number_of_rooms):
        r = {'x': Int('room_{}_x'.format(i)), 'y': Int('room_{}_y'.format(i)),
             'width': None, 'height': None, 'quad': None}
        if random.randint(1,100) <= EXCEPTION_RATE:
            r['width'] = random.randint(ROOM_WIDTH_MIN*4, ROOM_WIDTH_MAX*4)
            r['height'] = random.randint(ROOM_HEIGHT_MIN*4, ROOM_HEIGHT_MAX*4)
        else:
            r['width'] = random.randint(ROOM_WIDTH_MIN, ROOM_WIDTH_MAX)
            r['height'] = random.randint(ROOM_HEIGHT_MIN, ROOM_HEIGHT_MAX)
        #r['quad'] = i % 4 + 1
        r['quad'] = 1
        rooms.append(r)


def compute_room_centerpoints(m):
    cp = []

    for i in range(number_of_rooms):
        rooms[i]['center_x'] = m[rooms[i]['x']].as_long() + (rooms[i]['width']/2)
        rooms[i]['center_y'] = m[rooms[i]['y']].as_long() + (rooms[i]['height']/2)
        cp.append((rooms[i]['center_x'], rooms[i]['center_y']))
        #print("Centerpoint is: {}".format(cp))

    return cp


def create_big_room_constraints(slv):
    global and_clause_count, or_clause_count
    """ Make the first room 20% of size of playfield, and constrain its placement """
    # Throne room
    rooms[0]['width'] = int(0.4 * CANVAS_WIDTH)
    rooms[0]['height'] = int(0.6 * CANVAS_WIDTH * SCALE_FACTOR)

    slv.add(And(rooms[0]['x'] >= .3 * CANVAS_WIDTH, rooms[0]['x'] <= .35 * CANVAS_WIDTH,
                rooms[0]['y'] >= .1 * CANVAS_HEIGHT * SCALE_FACTOR,
                rooms[0]['y'] <= 0.25 * CANVAS_HEIGHT * SCALE_FACTOR))

    and_clause_count = and_clause_count + 4
    or_clause_count = or_clause_count + 0

    # Antechamber
    rooms[1]['width'] = int(0.4 * CANVAS_WIDTH)
    rooms[1]['height'] = int(0.05 * CANVAS_WIDTH * SCALE_FACTOR)

    rooms[2]['width'] = 15
    rooms[2]['height'] = 15 * SCALE_FACTOR


def create_separation_constraints(slv):
    global big_room_constraint

    for i in range(number_of_rooms):
        for j in range(i+1, number_of_rooms):
            if big_room_constraint:
                if i == 0 and (j == 1 or j == 2):
                    add_big_room_separation_constraint(slv, i, j)
                else:
                    add_separation_constraint(slv, i, j)
            else:
                add_separation_constraint(slv, i, j)


def add_big_room_separation_constraint(slv, i, j):
    global and_clause_count, or_clause_count
    # Have antechamber touching throne room bottom
    if j == 1:
        slv.add(And(rooms[i]['y'] == rooms[j]['y'] - rooms[i]['height'],
                    rooms[i]['x'] == rooms[j]['x']))
        and_clause_count = and_clause_count + 2
        or_clause_count = or_clause_count + 0
    # Have escape room touching top right of throne room
    if j == 2:
        slv.add(And(rooms[i]['x'] == rooms[j]['x'] - rooms[i]['width'],
                    rooms[i]['y'] == rooms[j]['y'] - 0.1 * rooms[i]['height']))
        and_clause_count = and_clause_count + 2
        or_clause_count = or_clause_count + 0

def add_separation_constraint(slv, i, j):
    global and_clause_count, or_clause_count
    vert_cond = {"above": "rooms[j]['y'] <= (rooms[i]['y'] - rooms[j]['height'] - SEPARATION_Y)",
#                 "same": "rooms[j]['y'] == rooms[i]['y']",
                 "same": None,
                 "below": "rooms[i]['y'] <= (rooms[j]['y'] - rooms[i]['height'] - SEPARATION_Y)" }

    horiz_cond = {"left": "rooms[j]['x'] <= (rooms[i]['x'] - rooms[j]['width'] - SEPARATION)",
#                  "same": "rooms[j]['x'] == rooms[i]['x']",
                  "same": None,
                  "right": "rooms[i]['x'] <= (rooms[j]['x'] - rooms[i]['width'] - SEPARATION)" }

    constraint = "Or("

    # for dir in directions:
    #     if vert_cond[dir['vert']] is not None and horiz_cond[dir['horiz']] is not None:
    #         constraint += "And(" + vert_cond[dir['vert']] + ", " + horiz_cond[dir['horiz']] + "),\n"
    #     if vert_cond[dir['vert']] is None and horiz_cond[dir['horiz']] is not None:
    #         constraint += horiz_cond[dir['horiz']] + ",\n"
    #     if vert_cond[dir['vert']] is not None and horiz_cond[dir['horiz']] is None:
    #         constraint += vert_cond[dir['vert']] + ",\n"
    constraint += vert_cond['above'] + ",\n" + vert_cond['below'] + ",\n" + horiz_cond['left'] + ",\n" + horiz_cond['right'] + ",\n"

    constraint = constraint[:-2]
    constraint += "\n)"
    slv.add(eval(constraint))
    and_clause_count = and_clause_count + 1
    or_clause_count = or_clause_count + 4


def create_canvas_constraints(slv):
    global and_clause_count, or_clause_count
    for i in range(number_of_rooms):
        slv.add(rooms[i]['x'] >= 0, rooms[i]['x'] + rooms[i]['width'] <= CANVAS_WIDTH)
        slv.add(rooms[i]['y'] >= 0, rooms[i]['y'] + rooms[i]['height'] <= CANVAS_HEIGHT * SCALE_FACTOR)
        and_clause_count = and_clause_count + 4
        or_clause_count = or_clause_count + 0


def create_line_constraints(slv):
    global and_clause_count, or_clause_count
    for i in range(number_of_rooms):
        # slv.add(
        #     Or(
        #     And(rooms[i]['y'] == rooms[i]['x'], rooms[i]['x'] <= 100)
        #     ,
        #     And(rooms[i]['y'] == 2*rooms[i]['x'], rooms[i]['x'] > 100)
        # )
        # )
        # Simple line
        #slv.add(And(rooms[i]['x'] >= rooms[i]['y']-LINEWIDTH, rooms[i]['x'] <= rooms[i]['y']+LINEWIDTH))

        # X shape
        slv.add(Or(
            And(rooms[i]['y'] + rooms[i]['x'] >= 380 - LINEWIDTH, rooms[i]['y'] + rooms[i]['x'] <= 380 + LINEWIDTH),
            And(rooms[i]['x'] >= rooms[i]['y'] - LINEWIDTH, rooms[i]['x'] <= rooms[i]['y'] + LINEWIDTH)
        ))
        and_clause_count = and_clause_count + 4
        or_clause_count = or_clause_count + 2

def create_point_line_constraints(slv, lines):
    global big_room_constraint, and_clause_count, or_clause_count

    for i in range(number_of_rooms):
        if big_room_constraint and (i == 0 or i == 1 or i == 2):
            continue

        constraint = "Or("
        for line in lines:
            # Separating numerator and denominator of slope causes significant slowdown, due to division
            #constraint += "((rooms[i]['y'] - " + str(line['y2']) +") == (" + str(line['m_num']) + " * (rooms[i]['x'] - " \
            #               + str(line['x2']) + ")) / " + str(line['m_den']) + "),\n"

            if line['m'] > 0:
                constraint += "And((rooms[i]['y'] <= " + str(line['m']) + "* (rooms[i]['x'] - " \
                               + str(line['x2']) + "+" + str(LINEWIDTH) + ") +" + str(line['y2']) + "),\n"
                constraint += "(rooms[i]['y'] >= " + str(line['m']) + "* (rooms[i]['x'] - " \
                               + str(line['x2']) + "-" + str(LINEWIDTH) + "+" + str(rooms[i]['width']) + ")+" + str(line['y2']) + "),\n"
            else:
                constraint += "And((rooms[i]['y'] >= " + str(line['m']) + "* (rooms[i]['x'] - " \
                               + str(line['x2']) + "+" + str(LINEWIDTH) + ") +" + str(line['y2']) + "),\n"
                constraint += "(rooms[i]['y'] <= " + str(line['m']) + "* (rooms[i]['x'] - " \
                               + str(line['x2']) + "-" + str(LINEWIDTH) + "+" + str(rooms[i]['width']) + ")+" + str(line['y2']) + "),\n"


            if line['y2'] > line['y1']:
                high_y = line['y2']
                low_y = line['y1']
            else:
                high_y = line['y1']
                low_y = line['y2']

            # check to see if y-height range is too small. If so, use x range instead
            if high_y - rooms[i]['height'] > low_y:
                # y range is fine
                constraint += "(rooms[i]['y'] >= " + str(low_y) + "),\n"
                constraint += "(rooms[i]['y'] <= " + str(high_y) + "-" + str(rooms[i]['height']) + ")),\n"
            else:
                # use x range
                if line['x2'] > line['x1']:
                    high_x = line['x2']
                    low_x = line['x1']
                else:
                    high_x = line['x1']
                    low_x = line['x2']
                constraint += "(rooms[i]['x'] >= " + str(low_x) + "),\n"
                constraint += "(rooms[i]['x'] <= " + str(high_x) + "-" + str(rooms[i]['width']) + ")),\n"

            and_clause_count = and_clause_count + 4
            or_clause_count = or_clause_count + 1

        constraint = constraint[:-2]
        constraint += "\n)"
        print("Room: {}  Constraint: \n{}\n\n".format(i,constraint))
        slv.add(eval(constraint))


def create_mousepoint_constraints(slv, mousepoints):
    """ Add a series of linear constraints following lines created by mousepoints """
    lines = []
    prev = None
    for p in mousepoints:
        if prev is None:
            prev = p
            continue
        x1 = (prev[0] - BORDER)
        y1 = (prev[1] - BORDER) * SCALE_FACTOR
        x2 = (p[0] - BORDER)
        y2 = (p[1] - BORDER) * SCALE_FACTOR
        # represent slope as a numerator and denominator, multiplied by 1000 (to integerize the floating point math)
        m_num = (y2 - y1)
        if (x2-x1) == 0:
            m_den = 1
        else:
            m_den = (x2 - x1)
        l_info = {'m_num': m_num, 'm_den': m_den, 'm': m_num/m_den, 'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2}
        #print("prev: {} || p: {}".format(prev,p))
        print("slope: {}  slope_num: {}  slope_den: {}  x2: {}  y2: {}".format(m_num/m_den, m_num, m_den, x2, y2))
        lines.append(l_info)
        prev = p

    create_point_line_constraints(slv, lines)


def create_quad_constraints(slv):
    global and_clause_count, or_clause_count

    for i in range(number_of_rooms):
        # upper left
        if rooms[i]['quad'] == 1:
            slv.add(And(rooms[i]['x'] <= ((CANVAS_WIDTH/2) - rooms[i]['width']), rooms[i]['y'] <= (CANVAS_HEIGHT/2)-rooms[i]['height']))
        # upper right
        if rooms[i]['quad'] == 2:
            slv.add(And(rooms[i]['x'] > CANVAS_WIDTH/2, rooms[i]['y'] <= (CANVAS_HEIGHT/2) - rooms[i]['height']))
        # lower left
        if rooms[i]['quad'] == 3:
            slv.add(And(rooms[i]['x'] <= ((CANVAS_WIDTH/2) - rooms[i]['width']), rooms[i]['y'] >= CANVAS_HEIGHT/2))
        # lower right
        if rooms[i]['quad'] == 4:
            slv.add(And(rooms[i]['x'] > CANVAS_WIDTH/2, rooms[i]['y'] >= CANVAS_HEIGHT/2))
        and_clause_count = and_clause_count + 2
        or_clause_count = or_clause_count + 0

def init_all_constraints(slv, mousepoints=None):
    global line_constraints
    global quad_constraints
    global big_room_constraint
    global timing_info
    global and_clause_count, or_clause_count

    and_clause_count = 0
    or_clause_count = 0
    begin = time.perf_counter()
    all_begin = begin
    create_canvas_constraints(slv)
    end = time.perf_counter()
    timing_info['create_canvas_constraints'] = end-begin

    if big_room_constraint:
        begin = time.perf_counter()
        create_big_room_constraints(slv)
        end = time.perf_counter()
        timing_info['create_big_room_constraints'] = end - begin

    begin = time.perf_counter()
    create_separation_constraints(slv)
    end = time.perf_counter()
    timing_info['create_separation_constraints'] = end - begin

    if line_constraints:
        begin = time.perf_counter()
        create_line_constraints(slv)
        end = time.perf_counter()
        timing_info['create_line_constraints'] = end - begin

    if quad_constraints:
        begin = time.perf_counter()
        create_quad_constraints(slv)
        end = time.perf_counter()
        timing_info['create_quad_constraints'] = end - begin

    if len(mousepoints) >= 2:
        begin = time.perf_counter()
        create_mousepoint_constraints(slv, mousepoints)
        end = time.perf_counter()
        timing_info['create_control_line_constraints'] = end - begin

    all_end = time.perf_counter()
    timing_info['create_all_constraints'] = all_end - all_begin
    print("======")
    print("And clause count: {}".format(and_clause_count))
    print("Or clause count: {}".format(or_clause_count))


def print_check(slv):
    print("Combined check")
    print(assumptions)
    print(slv.check(assumptions))

    for a in assumptions:
        print("{}: {}".format(a, slv.check(a)))


def print_model(m):
    for elem in m.decls():
        print("{} = {}".format(elem, m[elem]))


def display_room_info():
    for i in range(number_of_rooms):
        print("Room {}:: width: {}  height: {}".format(i, rooms[i]['width'], rooms[i]['height']))


def display_room_plus_model(m):
    for i in range(number_of_rooms):
        print("Room {}:: width: {}  height: {}  x: {}   y: {}".format(i, rooms[i]['width'], rooms[i]['height'],
                                                                      m[rooms[i]['x']], m[rooms[i]['y']]))


def draw_rooms(m, surf, tri=None, mst=None, cp=None, points=None):
    global show_sparse, show_delaunay
    surf.fill((255, 255, 255))
    r = None
    for i in range(number_of_rooms):
        r = pygame.Rect((m[rooms[i]['x']].as_long()+BORDER), (m[rooms[i]['y']].as_long())/SCALE_FACTOR+BORDER,
                        rooms[i]['width'], rooms[i]['height']/SCALE_FACTOR)
        if rooms[i]['quad'] == 1:  # White/Black
            pygame.draw.rect(surf, (0, 0, 0), r, 2)
        if rooms[i]['quad'] == 2:  # Orange
            pygame.draw.rect(surf, (255,133,27), r, 2)
        if rooms[i]['quad'] == 3:  # Blue
            pygame.draw.rect(surf, (0,116,217), r, 2)
        if rooms[i]['quad'] == 4:  #
            pygame.draw.rect(surf, (46,204,64), r, 2)

    if tri is not None:
        # Draw Delaunay triangulation
        for t in tri.simplices:
            #print("t is: {}".format(t))
            plist = []
            for e in t:
                #print("e is {}".format(e))
                plist.append((int(cp[e][0])+BORDER, int(cp[e][1]/SCALE_FACTOR)+BORDER))
                # print(t)
                # print(e)
                # print(type(e))
                #print(cp[t])
            #print("plist is: {}".format(plist))
            if show_delaunay:
                pygame.draw.aalines(surf, (0, 45, 225), True, plist)

    if mst is not None:
        # print("mst.data is: {}".format(mst.data))
        # print("mst.data type is: {}".format(type(mst.data)))
        # print("mst.indices is: {}".format(mst.indices))
        # print("mst.indices type is: {}".format(type(mst.indices)))
        # print("mst.indptr is: {}".format(mst.indptr))
        # print("mst.indptr type is: {}".format(type(mst.indptr)))
        #mst_list = [((i, j), mst[i,j]) for i, j in zip(*mst.nonzero())]

        for i, j in zip(*mst.nonzero()):
            #print("({},{}) = {}".format(i, j, mst[i,j]))
            plist = []
            plist.append((int(cp[i][0])+BORDER, int(cp[i][1]/SCALE_FACTOR)+BORDER))
            plist.append((int(cp[j][0])+BORDER, int(cp[j][1]/SCALE_FACTOR)+BORDER))
            if show_sparse:
                pygame.draw.aalines(surf, (0, 225, 0), True, plist)

    if points is not None:
        draw_lines(surf, points)

    draw_passageways(m, surf, mst)

    pygame.display.flip()


def distance(p1, p2):
    return math.sqrt(pow(p2[0]-p1[0], 2) + pow((p2[1]-p1[1])/SCALE_FACTOR, 2))


def create_graph_array(tri, cp):
    """ Given a Delaunay triangulation, creates a matrix form of this, with edge weights as lengths """
    graph = np.zeros((number_of_rooms, number_of_rooms))
    #print("graph is: {}".format(graph))

    if tri is not None:
        for t in tri.simplices:
            graph[t[0]][t[1]] = distance(cp[t[0]], cp[t[1]])
            graph[t[1]][t[2]] = distance(cp[t[1]], cp[t[2]])
            graph[t[2]][t[0]] = distance(cp[t[2]], cp[t[0]])

    #print("graph is: {}".format(graph))
    return graph


def draw_lines(surf, points):
    if len(points) < 2:
        return

    p = points[0]
    r = pygame.draw.aalines(surf, (139, 0, 0), False, points)
    pygame.display.flip()


def overlap(x, y):
    """ From: https://stackoverflow.com/questions/6821156/how-to-find-range-overlap-in-python """
    if not range_overlapping(x, y):
        return set()
    return set(range(max(x.start, y.start), min(x.stop, y.stop)+1))


def range_overlapping(x, y):
    """ From: https://stackoverflow.com/questions/6821156/how-to-find-range-overlap-in-python """
    if x.start == x.stop or y.start == y.stop:
        return False
    return ((x.start < y.stop  and x.stop > y.start) or
            (x.stop  > y.start and y.stop > x.start))


def draw_passageways(m, surf, mst):
    """ Draw passageways connecting rooms """
    for i, j in zip(*mst.nonzero()):
        plist = []
        # Determine which room is above the other room
        if m[rooms[i]['y']].as_long() < m[rooms[j]['y']].as_long():
            top = i
            bottom = j
        else:
            top = j
            bottom = i
        # Determine which room is to right of other room
        if m[rooms[i]['x']].as_long() < m[rooms[j]['x']].as_long():
            left = i
            right = j
        else:
            right = i
            left = j

        top_x_range = range(m[rooms[top]['x']].as_long()+int(PASSAGE_WIDTH), m[rooms[top]['x']].as_long()+rooms[top]['width']-int(PASSAGE_WIDTH))
        top_y_range = range(m[rooms[top]['y']].as_long()+int(PASSAGE_WIDTH), m[rooms[top]['y']].as_long()+rooms[top]['height']-int(PASSAGE_WIDTH))
        bottom_x_range = range(m[rooms[bottom]['x']].as_long()+int(PASSAGE_WIDTH), m[rooms[bottom]['x']].as_long()+rooms[bottom]['width']-int(PASSAGE_WIDTH))
        bottom_y_range = range(m[rooms[bottom]['y']].as_long()+int(PASSAGE_WIDTH), m[rooms[bottom]['y']].as_long()+rooms[bottom]['height']-int(PASSAGE_WIDTH))

        if range_overlapping(top_x_range, bottom_x_range):
            if range_overlapping(top_y_range, bottom_y_range):
                print("Rooms overlapping??")
            else:
                # x overlap, no y overlap. Drop passage down from top room to bottom room
                pass_x = random.choice(tuple(overlap(top_x_range, bottom_x_range)))
                plist.append((pass_x+BORDER, ((m[rooms[top]['y']].as_long()+rooms[top]['height'])/SCALE_FACTOR)+BORDER))
                plist.append((pass_x+BORDER, ((m[rooms[bottom]['y']].as_long()/SCALE_FACTOR)+BORDER)))
                pygame.draw.lines(surf, (0, 0, 0), False, plist, PASSAGE_WIDTH)
        else:
            if range_overlapping(top_y_range, bottom_y_range):
                # y overlap, no x overlap, draw line straight across
                pass_y = random.choice(tuple(overlap(top_y_range, bottom_y_range)))
                plist.append((m[rooms[left]['x']].as_long()+rooms[left]['width']+BORDER, ((pass_y/SCALE_FACTOR)+BORDER)))
                plist.append((m[rooms[right]['x']].as_long()+BORDER, ((pass_y/SCALE_FACTOR)+BORDER)))
                pygame.draw.lines(surf, (0, 0, 0), False, plist, PASSAGE_WIDTH)
            else:
                # no x overlap, no y overlap, draw right-angle connector
                # TODO: check for intersections
                pass_x = random.choice(bottom_x_range)
                pass_y = random.choice(top_y_range)
                if top == left:
                    plist.append((m[rooms[top]['x']].as_long()+rooms[top]['width']+BORDER, ((pass_y/SCALE_FACTOR)+BORDER)))
                else:
                    plist.append((m[rooms[top]['x']].as_long() + BORDER,
                                  ((pass_y / SCALE_FACTOR) + BORDER)))
                plist.append((pass_x+BORDER+PASSAGE_WIDTH/2, ((pass_y/SCALE_FACTOR)+BORDER)))
                pygame.draw.lines(surf, (0, 0, 0), False, plist, PASSAGE_WIDTH)
                plist=[]
                plist.append((pass_x + BORDER, ((pass_y / SCALE_FACTOR) + BORDER)))
                plist.append((pass_x+BORDER, ((m[rooms[bottom]['y']].as_long()/SCALE_FACTOR)+BORDER)))
                pygame.draw.lines(surf, (0, 0, 0), False, plist, PASSAGE_WIDTH)


def init_grid_counts():
    """ Initialize the grid used for computing rectangle placement density """
    global grid_counts

    grid_counts = [[0 for y in range(int(CANVAS_HEIGHT / GRID_CELL))]
                   for x in range(int(CANVAS_WIDTH / GRID_CELL))]


def update_grid(m):
    """ Update the grid counts based on the current set of rooms """
    for i in range(number_of_rooms):
        update_grid_counts(m, rooms[i])


def update_grid_counts(m, r):
    """ Update grid counts based on which cells the passed room overlaps """
    r_x = m[r['x']].as_long()
    r_y = m[r['y']].as_long()
    sx = int(r_x/GRID_CELL)
    # print("*** x: {}  GRID_CELL: {}  start_x: {}".format(r_x, GRID_CELL, sx))
    start_grid_x = int(r_x/GRID_CELL)
    end_grid_x = int((r_x+r['width'])/GRID_CELL)
    start_grid_y = int(r_y/GRID_CELL_Y)
    end_grid_y = int((r_y+r['height'])/GRID_CELL_Y)

    # print("==========")
    # print("Room: x: {}  width: {}  y: {}  height: {}".format(r_x, r['width'], r_y, r['height']))
    # print("grid: x: {}-{}  y: {}-{} ".format(start_grid_x, end_grid_x, start_grid_y, end_grid_y))

    for i in range(start_grid_y, end_grid_y):
        for j in range(start_grid_x, end_grid_x):
            # print("i: {}  j: {}".format(i, j))
            grid_counts[i][j] += 1


def make_heatmap():
    """ Plot a heatmap from grid count data """
    global number_of_rooms

    # Convert raw counts into averages
    for i in range(int(CANVAS_HEIGHT / GRID_CELL)):
        for j in range(int(CANVAS_WIDTH / GRID_CELL)):
            grid_counts[i][j] /= NUM_RUNS

    sns_plot = sns.heatmap(grid_counts, linewidth=0.1, square=True, vmin=0, vmax=80)
    fig = sns_plot.get_figure()
    fig.savefig("{}x{}_{}_{}_{}_{}_{}_{}rooms_{}runs.svg".format(CANVAS_WIDTH, CANVAS_HEIGHT, ROOM_WIDTH_MIN, ROOM_WIDTH_MAX,
                                               ROOM_HEIGHT_MIN, ROOM_HEIGHT_MAX, "quads" if quad_constraints else "noquads",
                                               number_of_rooms, NUM_RUNS), format="svg")
    fig.clf()


def reset_accumulated_timing():
    for timing in accumulated_timing_info:
        accumulated_timing_info[timing] = []


def update_timing():
    global timing_info
    global accumulated_timing_info

    if not accumulated_timing_info:
        print("Initializing accumulated_timing_info")
        for timing in timing_info:
            accumulated_timing_info[timing] = []

    for timing in timing_info:
        accumulated_timing_info[timing].append(timing_info[timing])


def save_accumulated_timing():
    global accumulated_timing_info
    global timing_per_run
    final_analysis = {}

    for timing in accumulated_timing_info:
        s_average = timing + "_average"
        s_median = timing + "_median"
        s_minimum = timing + "_minimum"
        s_maximum = timing + "_maximum"
        if timing_per_run.get(s_average) is None:
            timing_per_run.update({s_average: []})
        if timing_per_run.get(s_median) is None:
            timing_per_run.update({s_median: []})
        if timing_per_run.get(s_minimum) is None:
            timing_per_run.update({s_minimum: []})
        if timing_per_run.get(s_maximum) is None:
            timing_per_run.update({s_maximum: []})
        if timing_per_run.get(timing) is None:
            timing_per_run.update({timing: []})

        timing_per_run[timing].append(accumulated_timing_info[timing])
        print("===\nAccumulated_timing_info <<{}>>\n{}\n\n".format(timing,accumulated_timing_info[timing]))
        print("===\nTiming_per_run <<{}>>\n{}\n\n".format(timing, timing_per_run[timing]))

        avg = np.average(accumulated_timing_info[timing])
        final_analysis[timing+"_average"] = avg
        print("{} average: {}".format(timing, avg))
        #timing_per_run[s_average].append(avg)
        med = np.median(accumulated_timing_info[timing])
        final_analysis[timing+"_median"] = med
        print("{} median: {}".format(timing, med))
        #timing_per_run[s_median].append(med)
        min = np.amin(accumulated_timing_info[timing])
        final_analysis[timing+"_minimum"] = min
        print("{} minimum: {}".format(timing, min))
        #timing_per_run[s_minimum] = min
        max = np.amax(accumulated_timing_info[timing])
        final_analysis[timing+"_maximum"] = max
        print("{} maximum: {}".format(timing, max))
        #timing_per_run[s_maximum] = max

    reset_accumulated_timing()



def do_histogram(arr, datatype):
    sns_plot = sns.distplot(arr, kde=False, rug=True)
    fig = sns_plot.get_figure()
    fig.savefig("{}_hist_{}x{}_{}_{}_{}_{}_{}_{}rooms_{}runs.svg".format(datatype, CANVAS_WIDTH, CANVAS_HEIGHT, ROOM_WIDTH_MIN,
                                                                 ROOM_WIDTH_MAX,
                                                                 ROOM_HEIGHT_MIN, ROOM_HEIGHT_MAX,
                                                                 "quads" if quad_constraints else "noquads",
                                                                 number_of_rooms, NUM_RUNS), format="svg")
    fig.clf()


def final_data_analysis():
    final_analysis = {}

    for timing in accumulated_timing_info:
        print("================")
        avg = np.average(accumulated_timing_info[timing])
        final_analysis[timing+"_average"] = avg
        print("{} average: {}".format(timing, avg))
        #timing_per_run[s_average].append(avg)
        med = np.median(accumulated_timing_info[timing])
        print("{} median: {}".format(timing, med))
        final_analysis[timing+"_median"] = med
        #timing_per_run[s_median].append(med)
        min = np.amin(accumulated_timing_info[timing])
        print("{} minimum: {}".format(timing, min))
        final_analysis[timing+"_minimum"] = min
        #timing_per_run[s_minimum] = min
        max = np.amax(accumulated_timing_info[timing])
        print("{} maximum: {}".format(timing, max))
        final_analysis[timing+"_maxiumu"] = max
        #timing_per_run[s_maximum] = max
        do_histogram(accumulated_timing_info[timing], timing)

        filename = "{}_hist_{}x{}_{}_{}_{}_{}_{}_{}rooms_{}runs.json".format("Alldata", CANVAS_WIDTH, CANVAS_HEIGHT,
                                                                            ROOM_WIDTH_MIN,
                                                                            ROOM_WIDTH_MAX,
                                                                            ROOM_HEIGHT_MIN, ROOM_HEIGHT_MAX,
                                                                            "quads" if quad_constraints else "noquads",
                                                                            number_of_rooms, NUM_RUNS)
        myfile = open(filename, 'w')
        myfile.write(json.dumps(final_analysis))


def save_mousepoint_data(mp):
    myfile = open("mousepoints.json", 'w')
    myfile.write(json.dumps(mp))
    print("Wrote mousepoints to disk, file: mousepoints.json")

def load_mousepoint_data():
    myfile = open("mousepoints.json", 'r')
    mp = json.load(myfile)
    print("Read mousepoints from disk, file: mousepoints.json")
    print("{}".format(mp))
    return mp


def main():
    print("Dungeon layout using SMT")
    global number_of_rooms
    global show_sparse
    global show_delaunay
    global timing_info

    pygame.init()
    surface = pygame.display.set_mode((CANVAS_WIDTH+2*BORDER, CANVAS_HEIGHT+2*BORDER))

    loop_count = NUM_LOOPS
    run_count = NUM_RUNS
    unsat_count = 0
    looper = True
    run_once = False
    center_points = None
    init_grid_counts()
    tri = None
    mousepoints = []
    timing_info = {}

    while looper:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    init_rooms()
                    solver = Solver()
                    init_all_constraints(solver, mousepoints)
                    #
                    # begin = time.perf_counter()
                    # s = solver.check()
                    # end = time.perf_counter()
                    # print(s)
                    # print("Solve time: {} ms".format((end-begin)*1000.))
                    # timing_info['solve_time'] = end-begin
                    # if s.r == Z3_L_TRUE:
                    #     display_room_plus_model(solver.model())
                    #     begin = time.perf_counter()
                    #     center_points = compute_room_centerpoints(solver.model())
                    #     tri = Delaunay(center_points)
                    #     end = time.perf_counter()
                    #     timing_info['delaunay_time'] = end-begin
                    #     begin = time.perf_counter()
                    #     ar = create_graph_array(tri, center_points)
                    #     tcsr = minimum_spanning_tree(ar)
                    #     end = time.perf_counter()
                    #     timing_info['mst_time'] = end - begin
                    #     #print("min span tree: \n{}\n\n".format(tcsr.toarray().astype(int)))
                    #     draw_rooms(solver.model(), surface, tri, tcsr, center_points, mousepoints)
                    # else:
                    #     display_room_info()
                    run_once = True
                if event.key == pygame.K_ESCAPE:
                    looper = False
                if event.key == pygame.K_l:
                    line_constraints = not line_constraints
                if event.key == pygame.K_EQUALS:
                    number_of_rooms += 1
                if event.key == pygame.K_MINUS:
                    number_of_rooms -= 1
                if event.key == pygame.K_d:
                    show_delaunay = not show_delaunay
                if event.key == pygame.K_s:
                    show_sparse = not show_sparse
                if event.key == pygame.K_u:
                    save_mousepoint_data(mousepoints)
                if event.key == pygame.K_i:
                    mousepoints = load_mousepoint_data()
#                if event.key == pygame.K_z:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mousepoints.append(event.pos)
                    print("Added: {}".format(event.pos))
                    draw_lines(surface, mousepoints)


        if run_once:
            time.sleep(1)
            begin = time.perf_counter()
            s = solver.check()
            end = time.perf_counter()
            print(s)
            print("Solve time: {} ms".format((end-begin)*1000.))
            #print("statistics for the last check method...")
            # Traversing statistics
            #stat = solver.statistics()
            #print("Stat is: {}".format(stat))
            #for k, v in stat:
            #    print("{} : {}".format(k, v))
            asrts = solver.assertions()
            print("@@@ Assertions @@@")
            i = 0
            for asrt in asrts:
                i = i + 1
                #print("Assertion: {}".format(asrt))
            print("Total assertions: {}".format(i))
            timing_info['solve_time'] = end-begin
            if s.r == Z3_L_TRUE:
                #display_room_plus_model(solver.model())
                begin = time.perf_counter()
                center_points = compute_room_centerpoints(solver.model())
                tri = Delaunay(center_points)
                end = time.perf_counter()
                timing_info['delaunay_time'] = end - begin
                begin = time.perf_counter()
                ar = create_graph_array(tri, center_points)
                tcsr = minimum_spanning_tree(ar)
                end = time.perf_counter()
                timing_info['mst_time'] = end - begin
                draw_rooms(solver.model(), surface, tri, tcsr, center_points, mousepoints)
                update_grid(solver.model())
                loop_count -= 1
                update_timing()
                if loop_count <= 0:
                    #save_accumulated_timing()
                    run_count -= 1
                    print("#######\nRun: {}\n#######\n".format(run_count))
                    if run_count <= 0:
                        looper = False
                        make_heatmap()
                        final_data_analysis()
                    else:
                        init_rooms()
                        display_room_info()
                        solver = Solver()
                        init_all_constraints(solver, mousepoints)
                        loop_count = NUM_LOOPS
            else:
                print("Cannot find room placement!")
                display_room_info()
                init_rooms()
                loop_count = NUM_LOOPS
                unsat_count += 1
                if unsat_count > 10:
                    exit()



if __name__ == "__main__":
    main()
