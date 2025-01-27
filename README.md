Tested in julia 1.10.4
For current version see: 
https://julialang.org/downloads/

Building:
- In terminal:  'cd path_to_project/TransVis'
- run:  'julia --threads number_of_threads'
- type ']' to enter package manager
- In pkg mode: 'activate .' to activate TransVis Project 
- In pkg mode: 'instantiate' to install dependencies
- Backspace to exit pkg mode

In julia REPL (active TransVis Project)
- 'using TransVis' to precompile and export functions
- 'go()' is our current main function; needs trajectory name as a string e.g. `go("trajectory_name")`.

- first computation could take longer, as gradients need to be computed. Those will be stored for quick access in the "cache/" (Note that they are currently uniquely identified only by by their sequence. )
  

In code
- Pathes to Data are hardcoded in src/TransVis.jl , please ajust "stateDataPath", "sequence", and "transitionLabelData"
  ToDo: use JSON file to manage this








