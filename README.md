# smt_dungeon
Dungeon Layout Generator using SMT Solver

A program to create dungeon room layouts using the Z3 SMT (satisfiability modulo theories) constraint solver.

This is companion code for the paper, "Spatial Layout of Procedural Dungeons Using Linear Constraints and SMT Solvers", by Jim Whitehead, 
presented at the [11th Workshop on Procedural Content Generation (PCG 2020)](https://www.pcgworkshop.com/).

## Setup

This source code depends on:
* [Python Z3 bindings](https://github.com/Z3Prover/z3) (constraint solver)
* [Delaunay triangulation function in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html)
* [Minimum spanning tree function in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html)
* [Numpy](https://numpy.org/) (for statistical methods, not needed for core dungeon generation)
* [Seaborn](https://seaborn.pydata.org/) (for charts, not needed for core dungeon generation)
* [Matplot](https://matplotlib.org/) (for charts, not needed for core dungeon generation)
* [PyGame](https://www.pygame.org/) (for vizualization of dungeons, passages, and control lines)

Follow directions for each package to install.

## To Run

`python3 dungeon-smt.py`

This will cause a blank PyGame window to appear. Click inside the PyGame window so it receives keyboard events. Press `<space>` to begin dungeon execution. Statistics will be displayed to the console during execution. Press `<esc>` to exit. 

**Control Lines**

There are two ways to enter a control line.

1. After starting the generator, _before_ pressing `<space>`, use the mouse to click in the PyGame window. The first point will be invisible, but thereafter each point will define a new control line, displayed in red. Be careful that the control line is long enough so that there is enough space along the line to place the dungeon rooms (and thereby allow the Z3 solver to find layout solutions).
2. Load a control line file. After starting the generator and _before_ pressing `<space>`, press the `i` key. This will load a series of points from the file `mousepoints.json`. The points provided in the distributed `mousepoints.json` are the same points used in the control line examples in the paper.

**Altering Number of Rooms**

A thousand apologies, but this is quickly written research code which lacks obvious UI elements such as command line options. 

To change the number of rooms, at the top of the `dungeon-smt.py` file, change the constant `NUM_ROOMS`. As the number of rooms increases, it gets increasingly difficult for the Z3 solver to find a solution. Try small increments of 5 or 10 when you're first getting started, since determining that a solution isn't possible (determining unsat) can take a long time.

**Altering Room Sizes***

Room sizes are determined by a span of widths and heights. These are controlled by the following constants, located at the top of `dungeon-smt.py`:

*Width:* `ROOM_WIDTH_MIN` ... `ROOM_WIDTH_MAX`
*Height:* `ROOM_HEIGHT_MIN` ... `ROOM_HEIGHT_MAX`

Note that room heights have a scale factor element (e.g., `ROOM_HEIGHT_MIN = 20 * SCALE_FACTOR`). Only edit the `20`, and leave the `SCALE_FACTOR` alone. `SCALE_FACTOR` is used to get around the limitations of integer line slopes.

**Throne Room**

The paper discusses a constraint where there is a large throne room, with an adjacent antechamber and escape room. To turn on this scenario, edit the constant `big_room_constraint` to be `True` at the top of `dungeon-smt.py`.
