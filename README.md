Tested in julia 1.10.4
For current version see: 
https://julialang.org/downloads/

Building:
- In terminal:  'cd path_to_project/TransVis'
- run: `julia -p auto -i startup,jl`

In julia REPL (active TransVis Project)
- 'using TransVis' to precompile and export functions
- 'go()' is our current main function; needs trajectory name as a string e.g. `go("trajectory_name")`.

- first computation could take longer, as gradients need to be computed. Those will be stored for quick access in the "cache/" (Note that they are currently uniquely identified only by by their sequence. )
  

In code
- Pathes to Data are hardcoded in src/TransVis.jl , please ajust "stateDataPath", "sequence", and "transitionLabelData"
  ToDo: use JSON file to manage this


## Expected data format 
Inside the `data` directory, create a folder with a name that identifies the trajectory you're looking at. **The name of the folder will be used as an argument to the `go` function; i.e. `go("trajectory_name").** At a minimum, it needs the following files:
```
distances.pickle # atom-atom distance matrices per state; Dict{Int,Matrix}
transitions.pickle # list of transitions; Vector{Tuple{Int,Int}}
connectivity.pickle # atom-atom connectivity per state; Dict{Int, Matrix}
aligned_positions.pickle # positions per transition; Dict{Tuple{Int,Int}, Matrix}
```
You also need a folder called `dms`, with subfolders corresponding to distance matrices you're interested in. You need at least one for the program to start. Each distance matrix directory requires a file called `dm.pickle` containing a matrix / 2d array. **It is assumed that the rows of the distance matrix correspond to the transitions in the order presented by `transition.pickle`.**

You will also need an `alignment` folder containing pickles with dictionaries of tuples to matrices that will be used to perform intra-cluster alignments. Each matrix should be \[num_atoms * num_features\]. TransVis will calculate the center of mass for each feature and then align each transition using these centers of mass. We found the [bispectrum descriptor](https://www.nature.com/articles/s41524-022-00847-y) to be effective in aligning transitions, but in principle any descriptor can be used provided it returns the per-atom features in order.

You can optionally visualize per-atom scalars by placing dictionaries in the `scalars` folder. They must be dictionaries keyed by transition ids (i.e. (state1, state2)); the values of the dictionary must be a tuple of 1D arrays (Tuple{Vector{Float}, Vector{Float}}). 
