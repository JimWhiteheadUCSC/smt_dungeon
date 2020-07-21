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
2. Load a control line file. After starting the generator,_before_ pressing `<space>`, press the `i` key. This will load a series of points from the file `mousepoints.json`. The points provided in the distributed `mousepoints.json` are the same points used in the control line examples in the paper.

