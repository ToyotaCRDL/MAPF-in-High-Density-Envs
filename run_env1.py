import random

from solver.phans_null_agent_swapping import PHANS

def main():                
    random.seed(168)

    # --------------------------
    # problem definition
    # --------------------------
    size_x = 14
    size_y = 7
    
    static_obss = []

    task_lst = [(0, 0), (size_x-1, 0)]    
    
    tgt_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+task_lst))
    tgt_ag_start_pos_lst = random.sample(tgt_ag_option_nodes, len(task_lst))

    other_ag_option_nodes = list(set([(x, y) for x in range(size_x) for y in range(size_y)]) - set(static_obss+tgt_ag_start_pos_lst))
    other_ag_start_pos_lst = random.sample(other_ag_option_nodes, 88)

    ag_start_pos_lst = tgt_ag_start_pos_lst + other_ag_start_pos_lst

    # --------------------------
    # Run
    # --------------------------
    problem = PHANS(size_x, size_y, static_obstacles=static_obss)
    all_path_lst = problem.run_loop(ag_start_pos_lst, task_lst)
    print(f"makespan: {len(all_path_lst)}")
    
    # --------------------------
    # Save animation
    # --------------------------
    animation_name = f'{size_x}x{size_y}_{len(static_obss)}obs_{len(ag_start_pos_lst)}ags_{len(task_lst)}tgts_{len(all_path_lst)}steps.mp4'
    problem.plot_animation(animation_name, all_path_lst, small_env=True)

if __name__ == '__main__':
    main()
