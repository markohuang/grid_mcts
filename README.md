# grid mcts
A working implementation of the AlphaDev MCTS paper for a grid reconfiguration game.

## game description
1. game is initialized with a grid of size `n x n` (i.e., 2x6)
2. a set of objects are placed on the grid at random locations with unique labels (i.e., 0-8)

e.g., an example grid setup:
```
      2       1   8   6
  7   5   3       4   0
  ```
3. a game is comprised of turns where at the end of each turn, a set of tasks are performed on the grid 
4. tasks are specified as movement tuples where object 1 will be moved in a straight line to the location of object 2 before moving straight back to its original location. For example, the task `{(1,3),(8,4)}` is the set of movments `(1,3)` and `(8,4)` that entail moving object 1 to the location of object 3 and back, and moving object 8 to the location of object 4 and back. Note the movements can be performed in either order
6. the goal is to parallelize the movements of a task under the constraint that movements should not cross. For example, under the example grid the task `{(1,3),(8,4)}` is already parallelizable, but the task `{(6,4),(8,0)}` is not optimized because the two movements cross paths and would benefit from a reconfiguration that swaps object 8 and object 6
7. before each turn ends, the agent is allowed to reconfigure the grid, by sequentailly moving objects to empty spots on the grid. The step reward is then the number of task movements that are now parallelizable
8. the goal of the game is to parallelize as many tasks as possible with the fewest number of reconfigurations moves


## example result
![2x6 grid](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Example grid game played by MCTS agent")
