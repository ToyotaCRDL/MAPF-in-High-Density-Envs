# MAPF-HD: Multi-Agent Path Finding in High-Density Environments

This repository provides the official implementation of MAPF-HD, a framework for multi-agent path finding in high-density grid environments.


<div align="center">
<img src="animations/35x21_90obs_620ags_12tgts_143steps.gif" width="800">
</div>

## Background
Multi-agent path finding (MAPF) involves planning efficient paths for multiple agents to move simultaneously while avoiding collisions.
In typical warehouse environments, agents are often sparsely distributed along aisles.
However, increasing the agent density can improve space efficiency.
When the agent density is high, we must optimize the paths not only for goal-assigned agents but also for those obstructing them. 
This study proposes a novel MAPF framework for high-density environments (MAPF-HD). 
Several studies have explored MAPF in similar settings using integer linear programming (ILP). 
However, ILP-based methods require substantial computation time to optimize all agent paths simultaneously. 
Even in small grid-based environments with fewer than $100$ cells, these computations can incur tens to hundreds of seconds. 
These high computational costs render these methods impractical for large-scale applications such as automated warehouses and valet parking. 
To address these limitations, we introduce the phased null-agent swapping (PHANS) method. 
PHANS employs a heuristic approach to incrementally swap positions between agents and empty vertices. 
This method solves the MAPF-HD problem within seconds to tens of seconds, even in large environments containing more than $700$ cells. 
The proposed method can potentially improve efficiency in various real-world applications such as warehouse logistics, traffic management, or crowd control.


## Preparation

1. Clone this repository

2. Run
```
conda env create -f environment.yml
conda activate phans
```
- MP4 output requires ffmpeg to be available in the environment or on your system PATH.
- GIF output requires pillow.


## Minimal Usage
```
from solver.phans_null_agent_swapping import PHANS

size_x = 14
size_y = 7

# Static obstacles: list of (x, y)
static_obss = [(2, 3), (11, 3)] 

# Start positions for all agents: list of (x, y)
ag_start_pos_lst = [(12, 6), (1, 4),(1, 2), (4, 5), (5, 0), (5, 3), 
                    (6, 0), (8, 2), (9, 1), (11, 6), (12, 0), (13, 3)]

# Goal positions for target agents
# In this example, the first agent moves to (0, 0), the second to (13, 0).
task_lst = [(0, 0), (13, 0)]     

# Plan & execute
problem = PHANS(size_x, size_y, static_obstacles=static_obss)
all_path_lst = problem.run_loop(ag_start_pos_lst, task_lst)
print(f"makespan: {len(all_path_lst)}")

# Save animation (MP4 requires ffmpeg; use .gif to save a GIF via Pillow)
problem.plot_animation('animation.mp4', all_path_lst, small_env=True)
```


## Example

We provide the exact environments used in the paper's experiments.
The animation shown above corresponds to `run_env4.py`.
```
python run_env1.py
python run_env2.py
python run_env3.py
python run_env4.py
```


## Citation
If this code is useful for your work, please cite our paper:

```
@article{makino2025,
  title={MAPF-HD: Multi-Agent Path Finding in High-Density Environments},
  author={Makino, H. and Ito, S.},
  journal={arXiv preprint arXiv:2509.06374},
  year={2025}
}
```


## License
This repository is licensed under `LICENSE.md`.
