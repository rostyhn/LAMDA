# TransVis

## Setup 
To set up your julia environment, run `julia -t [num_workers],1 -i startup,jl`; `num_workers` should be your machine's number of processors - 1. To start the program, run `go("trajectory_name")` which should correspond to a name of a folder inside `data` e.g. `go('nano_pt')`.
  

## Expected data format 
Inside the `data` directory, create a folder with a name that identifies the trajectory you're looking at. **The name of the folder will be used as an argument to the `go` function; i.e. `go("trajectory_name").** The following is an example data directory: 
```
trajectory_name/ # used as input to go()
    distances.pickle # atom-atom distance matrices per state; Dict{Int,Matrix{Float}}
    transitions.pickle # list of transitions; Vector{Tuple{Int,Int}}
    connectivity.pickle # atom-atom connectivity per state; Dict{Int, Matrix{Float}}
    aligned_positions.pickle # positions per transition; Dict{Tuple{Int,Int}, Matrix{Float}}
    
    dms/ 
        - some_distance_metric/
            - dm.pickle # distance matrix defined for all transitions; Matrix{Float}
    alignment/
        - some_features.pickle # alignment features; Dict{Tuple{Int, Int}, Matrix{Float}}
    scalars/ # optional
        - some_scalar.pickle # per-atom scalar values; Dict{Tuple{Int,Int}, Tuple{Vector{Float},Vector{Float}}}

```
You also need a folder called `dms`, with subfolders corresponding to distance matrices you're interested in. You need at least one for the program to start. Each distance matrix directory requires a file called `dm.pickle` containing a matrix / 2d array. **It is assumed that the rows of the distance matrix correspond to the transitions in the order presented by `transition.pickle`.**

You will also need an `alignment` folder containing pickles with dictionaries of tuples to matrices that will be used to perform intra-cluster alignments with a variant of the [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm). Each matrix should be \[num_atoms * num_features\]. TransVis will calculate the center of mass for each feature and then align each transition using these centers of mass. We found the [bispectrum descriptor](https://www.nature.com/articles/s41524-022-00847-y) to be effective in aligning transitions, but in principle any descriptor can be used provided it returns the per-atom features in order.

You can optionally visualize per-atom scalars by placing dictionaries in the `scalars` folder. They must be dictionaries keyed by transition ids (i.e. (state1, state2)); the values of the dictionary must be a tuple of 1D arrays (Tuple{Vector{Float}, Vector{Float}}).
