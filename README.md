# TransVis

## Setup 
You need to install the Python environment manager `poetry`([https://python-poetry.org/]); once it is installed, run `poetry install`. 

To set up your Julia environment, run `./run.sh [num_workers]`; `num_workers` should be your machine's number of processors - 1. To start the program, run `go("trajectory_name")` which should correspond to a name of a folder inside `data` e.g. `go('nano_pt')`. 

`go` has some keyword parameters as well. 
| parameter | type | purpose |
| `chunk_size` | `Int` | sets how many volumes get processed at a time |
| `init_h_cutoff` | `Float` | sets the initial height cutoff value for the clustering |
| `align_with` | `String` | sets the initial feature values used to align the transitions |
| `distance_matrix` | `String` | sets which distance matrix to initially render |

## Expected data format 
Inside the `data` directory, create a folder with a name that identifies the trajectory you're looking at. **The name of the folder will be used as an argument to the `go` function; i.e. `go("trajectory_name").** The following is an example data directory: 
```
trajectory_name/ # used as input to go()
    distances.pickle # atom-atom distance matrices per state; Dict{Int,Matrix{Float}}
    transitions.pickle # list of transitions; Vector{Tuple{Int,Int}}
    connectivity.pickle # atom-atom connectivity per state; Dict{Int, Matrix{Float}}
    aligned_positions.pickle # positions per transition; Dict{Tuple{Int,Int}, Matrix{Float}}
    t_ase_dict.pickle # ASE data per transition; Dict{Tuple{Int,Int}, Tuple{Atoms, Atoms}}

    dms/ 
        - some_distance_metric/
            - dm.pickle # distance matrix defined for all transitions; Matrix{Float}
    alignment/
        - some_features.pickle # alignment features; Dict{Tuple{Int, Int}, Tuple{Matrix{Float}, Matrix{Float}}
    scalars/ # optional
        - some_scalar.pickle # per-atom scalar values; Dict{Tuple{Int,Int}, Tuple{Vector{Float},Vector{Float}}}
    per_t_scalars/ # optional
        - some_scalar.pickle # per-transition scalar values; Dict{Tuple{Int,Int}, Float}}

```
You also need a folder called `dms`, with subfolders corresponding to distance matrices you're interested in. You need at least one for the program to start. Each distance matrix directory requires a file called `dm.pickle` containing a matrix / 2d array. **It is assumed that the rows of the distance matrix correspond to the transitions in the order presented by `transition.pickle`.**

You will also need an `alignment` folder containing pickles with dictionaries of tuples to tuples of matrices that will be used to perform intra-cluster alignments with a variant of the [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm). Since each matrix should have signs, the initial state should have positive values & the final negative. Each matrix should be \[num_atoms * num_features\]. TransVis will calculate the center of charge for each feature and then align each transition using these centers of charge. We found the [bispectrum descriptor](https://www.nature.com/articles/s41524-022-00847-y) to be effective in aligning transitions, but in principle any descriptor can be used provided it returns the per-atom features in order.

You can optionally visualize per-atom scalars by placing dictionaries in the `scalars` folder and per-transition scalars in the `per_t_scalars`. The per-atom scalars must be dictionaries keyed by transition ids (i.e. (state1, state2)); the values of the dictionary are 1D arrays corresponding to each atom (Vector{Float}). `per_t_scalars` is accessed the same way, but contains per transition values.
