import copy

import matplotlib.animation as anm
import matplotlib.pyplot as plt
import numpy as np

from solver.a_star import AStarFollowingConflict


class PHANS:
    def __init__(
        self, 
        size_x: int, 
        size_y: int, 
        static_obstacles: list[tuple[int, int]] = None
    ):
        """
        A grid environment where multiple agents move while avoiding static and
        dynamic obstacles. This class plans target paths and executes stepwise
        movements, coordinating with "null" agents that vacate blocking cells.

        Args:
            size_x (int): Grid width.
            size_y (int): Grid height.
            static_obstacles (list[tuple[int, int]] | None): Coordinates that are
                permanently blocked.
        """
        if static_obstacles is None:
            static_obstacles = []

        self.size_x = size_x
        self.size_y = size_y
        self.static_obss = static_obstacles

        self.ag_start_pos_lst = []
        self.ag_cur_pos_lst = []
        self.null_ag_pos_lst = []
        self.all_posible_pos_lst = [
            (x, y)
            for x in range(self.size_x)
            for y in range(self.size_y)
            if (x, y) not in self.static_obss
        ]

        self.task_lst = []
        self.task_len = None

        self.tgt_ag_path_lst = []
        self.tgt_ag_path_lst_plot = []
        self.blocked_num_on_ag_path_lst = []
        self.num_of_obs_agents_on_path = []

    def _update_null_ag_pos_lst(self) -> None:
        """Update the list of positions currently available to null agents."""
        self.null_ag_pos_lst = list(
            set(self.all_posible_pos_lst) - set(self.ag_cur_pos_lst)
        )

    def _get_closest_null_ag_pos(
        self, 
        key_pos: tuple[int, int]
    ) -> None:
        """Return index and position of the closest null agent to key_pos."""
        self._update_null_ag_pos_lst()
        distances = np.abs(np.array(self.null_ag_pos_lst) - np.array(key_pos)).sum(
            axis=1
        )
        min_index = np.argmin(distances)
        return min_index, self.null_ag_pos_lst[min_index]

    def _update_goal_ag(self) -> None:
        """Update indices of agents that have reached their goals."""
        self.goal_ag_idx_lst = [
            i_tgt_ag
            for i_tgt_ag in range(self.task_len)
            if self.tgt_cur_steps[i_tgt_ag]
            == len(self.tgt_ag_path_lst[i_tgt_ag]) - 1
        ]

    def _is_num_of_obs_agents_on_path_increased(self) -> bool:
        """
        Check whether the number of obstructing agents along each target path
        increased compared to the previous check.
        """
        num_prev = copy.deepcopy(self.num_of_obs_agents_on_path)

        num_new = []
        for i_tgt in range(len(self.tgt_ag_path_lst)):
            num_of_obs_agents = 0
            for i in range(self.tgt_cur_steps[i_tgt], len(self.tgt_ag_path_lst[i_tgt]) - 1):
                if self.tgt_ag_path_lst[i_tgt][i] in self.ag_cur_pos_lst:
                    num_of_obs_agents += 1
            num_new.append(num_of_obs_agents)

        if len(num_prev) == 0:
            self.num_of_obs_agents_on_path = copy.deepcopy(num_new)
            return False

        for i in range(len(num_prev)):
            if num_prev[i] < num_new[i]:
                self.num_of_obs_agents_on_path = copy.deepcopy(num_new)
                return True

        self.num_of_obs_agents_on_path = copy.deepcopy(num_new)
        return False

    def _search_target_paths(
        self, 
        ags: list[dict], 
        a_star_max_iter: int = -1
    ) -> tuple[bool, list]:
        """
        Run PP to find paths for target agents.
        """
        dimension = (self.size_x, self.size_y)
        mov_obss = []
        mov_obss_edges = []
        solution_dict_all = [None for _ in range(self.task_len)]

        print("tgt_ag_path_lst: --------------------------")
        for ag in ags:
            other_ag_starts_moc_obss = [
                (a["start"][0], a["start"][1], t)
                for t in [0, 1]
                for a in ags
                if a != ag
            ]
            other_ag_goals = [a["goal"] for a in ags if a != ag]
            env = AStarFollowingConflict(
                dimension,
                ag,
                set(self.static_obss + other_ag_goals),
                moving_obstacles=mov_obss + other_ag_starts_moc_obss,
                moving_obstacle_edges=mov_obss_edges,
                a_star_max_iter=a_star_max_iter,
                agent_start_pos_lst=self.ag_start_pos_lst,
                null_agent_pos_lst=self.null_ag_pos_lst,
                considering_cycle_conflict=False,
            )
            solution = env.compute_solution()

            if len(solution) == 0:
                print("No solution found at init")
                return False, ag

            mov_obss_tmp = []

            mov_obss_tmp += [(s["x"], s["y"], s["t"]) for s in solution]
            solution_dict_all[ag["name"]] = solution
            print(solution)

            for i in range(len(mov_obss_tmp) - 1):
                mov_obss_edges.append(
                    (
                        mov_obss_tmp[i][0],
                        mov_obss_tmp[i][1],
                        mov_obss_tmp[i + 1][0],
                        mov_obss_tmp[i + 1][1],
                    )
                )
            mov_obss += mov_obss_tmp

        return True, solution_dict_all

    def _init_loop(
        self, 
        ag_start_pos_lst: list[tuple[int, int]], 
        task_lst: list[tuple[int, int]], 
        a_star_max_iter: int = -1
    ) -> None:
        """
        Initialization loop:
        - Relax conflicts and run target path search (PP).
        """
        self.task_lst = task_lst
        self.task_len = len(task_lst)
        self.ag_start_pos_lst = ag_start_pos_lst
        self.ag_cur_pos_lst = ag_start_pos_lst

        self.last_considered_blocking_ag_pos_lst = copy.deepcopy(
            self.ag_cur_pos_lst[: self.task_len]
        )
        self.blocked_num_on_ag_path_lst = [float("inf") for _ in range(self.task_len)]

        # PP
        self._update_null_ag_pos_lst()
        ags = [
            {"start": ag_start_pos_lst[i], "goal": self.task_lst[i], "name": i}
            for i in range(self.task_len)
        ]

        # Process in descending order of distance from start to goal
        dst_from_ag_to_goal_lst = [
            abs(ag_start_pos_lst[i][0] - self.task_lst[i][0])
            + abs(ag_start_pos_lst[i][1] - self.task_lst[i][1])
            for i in range(self.task_len)
        ]
        ags = [
            a
            for a, _ in sorted(
                zip(ags, dst_from_ag_to_goal_lst), key=lambda x: x[1], reverse=True
            )
        ]

        flag, solution_dict_all = self._search_target_paths(ags, a_star_max_iter)
        while not flag:
            # If pathfinding fails for an agent, prioritize it in the next attempt.
            failed_ag = solution_dict_all
            ags = [failed_ag] + [a for a in ags if a != failed_ag]
            flag, solution_dict_all = self._search_target_paths(ags, a_star_max_iter)

        self.tgt_ag_path_lst = []
        for i_ag in range(self.task_len):
            path_tmp = []
            if solution_dict_all[i_ag] is not None:
                for t in range(len(solution_dict_all[i_ag])):
                    path_tmp.append(
                        (solution_dict_all[i_ag][t]["x"], solution_dict_all[i_ag][t]["y"])
                    )
            self.tgt_ag_path_lst.append(path_tmp)

        print("---------------------------------------------")

    def run_loop(
        self, 
        ag_start_pos_lst: list[tuple[int, int]], 
        task_lst: list[tuple[int, int]], 
        a_star_max_iter: int = -1, 
        max_loop: int = 300
    ) -> list[list[tuple[int, int]]]:
        """
        Move agents step-by-step following the target paths.

        This loop:
        - Advances target agents along their target paths.
        - Detects following conflicts (including edge conflicts) and vertex
          conflicts against agents that already reached their goals.
        - Identifies obstructing agents and moves null agents to vacate those cells.
        """
        # -----------------------------------
        # 1. Plan target paths
        # -----------------------------------
        self._init_loop(ag_start_pos_lst, task_lst, a_star_max_iter)

        # -----------------------------------
        # 2. Evacuate obstructing agents and move target agents
        # -----------------------------------
        
        # Initialize
        all_path_lst = [copy.deepcopy(self.ag_cur_pos_lst)]
        self.tgt_cur_steps = [0] * len(self.tgt_ag_path_lst)

        loop = 0
        self._update_goal_ag()
        tgt_ags_cur_pos_lst = self.ag_cur_pos_lst[: self.task_len]
        
        for i_tgt_ag in [i for i in range(self.task_len) if i not in self.goal_ag_idx_lst]:
            self.tgt_cur_steps[i_tgt_ag] = (
                len(self.tgt_ag_path_lst[i_tgt_ag])
                - self.tgt_ag_path_lst[i_tgt_ag][::-1].index(tgt_ags_cur_pos_lst[i_tgt_ag])
                - 1
            )

        # Start loop
        while (
            any(
                self.tgt_cur_steps[i] < len(self.tgt_ag_path_lst[i]) - 1
                for i in range(self.task_len)
            )
            and loop < max_loop
        ):
            # Identify obstructing agents
            
            # next blocking agents on target paths
            blocking_ag_pos_lst = [] 
            # their remaining distance along the blocked target’s path to its goal
            next_blocking_ag_dist_from_goal_lst = []

            for i_tgt_ag in range(self.task_len):
                if self.tgt_cur_steps[i_tgt_ag] < len(self.tgt_ag_path_lst[i_tgt_ag]) - 1:
                    next_tgt_pos_lst = self.tgt_ag_path_lst[i_tgt_ag][
                        self.tgt_cur_steps[i_tgt_ag] + 1 :
                    ]

                    # Get the next blocking agent position
                    for pos in next_tgt_pos_lst:
                        if pos in self.ag_cur_pos_lst and pos not in tgt_ags_cur_pos_lst:
                            dst_to_goal = (
                                len(self.tgt_ag_path_lst[i_tgt_ag])
                                - self.tgt_ag_path_lst[i_tgt_ag].index(pos)
                                + 1
                            )
                            if pos not in blocking_ag_pos_lst:
                                blocking_ag_pos_lst.append(pos)
                                next_blocking_ag_dist_from_goal_lst.append(dst_to_goal)
                            else:
                                idx = blocking_ag_pos_lst.index(pos)
                                if next_blocking_ag_dist_from_goal_lst[idx] < dst_to_goal:
                                    # If multiple target agents share the same blocker,
                                    # increase its priority by updating with the larger distance.
                                    next_blocking_ag_dist_from_goal_lst[idx] = dst_to_goal

            # Order the obstructing agents
            blocking_ag_pos_lst = [
                pos
                for _, pos in sorted(
                    zip(
                        next_blocking_ag_dist_from_goal_lst,
                        blocking_ag_pos_lst,
                    ),
                    key=lambda x: x[0],
                    reverse=True,
                )
            ]

            # Define dependency among target agents:
            # If a target agent appears earlier on another's path, move the earlier one first.
            high_priority_blocking_ag_pos_lst = []
            dependency_relation = []  # (A, B): A depends on B

            for i_tgt_ag in [i for i in range(self.task_len) if i not in self.goal_ag_idx_lst]:
                tgt_ag_next_pos = self.tgt_ag_path_lst[i_tgt_ag][self.tgt_cur_steps[i_tgt_ag] + 1]
                tgt_ag_cur_idx = self.tgt_cur_steps[i_tgt_ag]

                for j_tgt_ag in range(self.task_len):
                    if i_tgt_ag != j_tgt_ag:
                        from_idx = self.tgt_cur_steps[j_tgt_ag]
                        to_idx = min(len(self.tgt_ag_path_lst[j_tgt_ag]) - 1, tgt_ag_cur_idx)
                        if tgt_ag_next_pos in self.tgt_ag_path_lst[j_tgt_ag][from_idx:to_idx]:
                            if j_tgt_ag in [x for x, _ in dependency_relation]:
                                dependency_relation.append((i_tgt_ag, j_tgt_ag))
                            else:
                                dependency_relation = [(i_tgt_ag, j_tgt_ag)] + dependency_relation

            for i_tgt_ag, j_tgt_ag in dependency_relation:
                tgt_ag_next_pos = self.tgt_ag_path_lst[i_tgt_ag][self.tgt_cur_steps[i_tgt_ag] + 1]
                from_idx = self.tgt_cur_steps[j_tgt_ag]
                high_priority_blocking_ag_pos_lst += self.tgt_ag_path_lst[j_tgt_ag][
                    from_idx : min(
                        len(self.tgt_ag_path_lst[j_tgt_ag]),
                        self.tgt_ag_path_lst[j_tgt_ag].index(tgt_ag_next_pos) + 2,
                    )
                ][::-1]

            for pos in high_priority_blocking_ag_pos_lst:
                if pos in blocking_ag_pos_lst:
                    blocking_ag_pos_lst.remove(pos)
                    blocking_ag_pos_lst = [pos] + blocking_ag_pos_lst

            # Choose the empty vertices
            self._update_null_ag_pos_lst()
            available_null_ags_pos_lst = copy.deepcopy(list(set(self.null_ag_pos_lst)))
            selected_null_ags_pos_lst = []

            for i, bag_pos in enumerate(blocking_ag_pos_lst):
                # Generally choose the null agent with the smallest h-value,
                # but avoid interfering with target agents' movements.

                dst_from_bag_to_goal = float("inf")
                for i_tgt_ag in range(self.task_len):
                    if bag_pos in self.tgt_ag_path_lst[i_tgt_ag]:
                        dst_from_bag_to_goal = min(
                            dst_from_bag_to_goal,
                            len(self.tgt_ag_path_lst[i_tgt_ag])
                            - self.tgt_ag_path_lst[i_tgt_ag].index(bag_pos)
                            - 1,
                        )

                # Preserve vertices between the target and obstructing agents on target paths
                tgt_preserved_pos = []
                dst_penalty_dct = {}

                for i_tgt_ag in range(self.task_len):
                    tgt_idx_on_tgt_ag_path = self.tgt_cur_steps[i_tgt_ag]

                    if bag_pos in self.tgt_ag_path_lst[i_tgt_ag]:
                        bag_idx_on_tgt_ag_path = self.tgt_ag_path_lst[i_tgt_ag].index(bag_pos)
                        for pos in self.tgt_ag_path_lst[i_tgt_ag][bag_idx_on_tgt_ag_path:]:
                            dst_from_null_to_tgt = np.sum(
                                np.abs(np.array(pos) 
                                       - np.array(self.tgt_ag_path_lst[i_tgt_ag]\
                                           [self.tgt_cur_steps[i_tgt_ag]])
                                )
                            )
                            if pos in dst_penalty_dct.keys():
                                dst_penalty_dct[pos] = max(
                                    dst_penalty_dct[pos],                        
                                    dst_from_null_to_tgt
                                )
                            else:
                                dst_penalty_dct[pos] = dst_from_null_to_tgt
                                
                    # For null agents from the target's current position to the goal:
                    for pos in self.tgt_ag_path_lst[i_tgt_ag][tgt_idx_on_tgt_ag_path:]:
                        dst_to_goal = (
                            len(self.tgt_ag_path_lst[i_tgt_ag])
                            - self.tgt_ag_path_lst[i_tgt_ag].index(pos)
                            - 1
                        )
                        if dst_from_bag_to_goal < dst_to_goal:
                            tgt_preserved_pos.append(pos)

                tmp_available_null_ags_pos_lst = [
                    null_ag_pos
                    for null_ag_pos in available_null_ags_pos_lst
                    if null_ag_pos not in tgt_preserved_pos
                ]

                if len(tmp_available_null_ags_pos_lst) > 0:
                    dst_penalty_array = np.array(
                        [dst_penalty_dct.get(pos, 0) for pos in tmp_available_null_ags_pos_lst]
                    )
                    distances = (
                        np.abs(np.array(tmp_available_null_ags_pos_lst) - np.array(bag_pos))
                        .sum(axis=1)
                    ) + dst_penalty_array
                    min_index = np.argmin(distances)
                    pos = tmp_available_null_ags_pos_lst[min_index]
                    selected_null_ags_pos_lst.append(pos)
                    available_null_ags_pos_lst.remove(pos)
                else:
                    break

            # Plan paths for null agents to move to obstructing agents' positions.
            dimension = (self.size_x, self.size_y)
            obss = self.static_obss + list(
                set(tgt_ags_cur_pos_lst)
                - set(selected_null_ags_pos_lst)
                - set(blocking_ag_pos_lst)
            )
            ags = []
            for i in range(len(selected_null_ags_pos_lst)):
                if len(ags) == 0 or blocking_ag_pos_lst[i] not in [ag["goal"] for ag in ags]:
                    ags.append(
                        {
                            "start": selected_null_ags_pos_lst[i],
                            "goal": blocking_ag_pos_lst[i],
                            "name": i,
                        }
                    )

            mov_obss = []
            solution_dict_all = []
            for ag in ags:
                tgt_preserved_pos = []
                for i_tgt_ag in range(self.task_len):
                    if ag["goal"] in self.tgt_ag_path_lst[i_tgt_ag]:
                        tgt_preserved_pos += [
                            pos
                            for pos in self.tgt_ag_path_lst[i_tgt_ag][
                                self.tgt_cur_steps[i_tgt_ag] : self.tgt_ag_path_lst[
                                    i_tgt_ag
                                ].index(ag["goal"])
                            ]
                        ]

                env = AStarFollowingConflict(
                    dimension,
                    ag,
                    set(obss + tgt_preserved_pos),
                    moving_obstacles=mov_obss,
                    moving_obstacle_edges=[],
                    a_star_max_iter=10000,
                    is_dst_add=False,
                )
                solution = env.compute_solution()
                mov_obss += [(s["x"], s["y"], s["t"]) for s in solution]
                solution_dict_all.append(solution)

            null_ags_path_lst = []
            for solution_dict in solution_dict_all:
                null_ags_path_lst.append([(s['x'], s['y']) for s in solution_dict])

            # Execute movements
            ag_next_pos_lst = copy.deepcopy(self.ag_cur_pos_lst)
            ag_pre_pos_lst = []

            one_tgt_ag_goal_flag = True
            while one_tgt_ag_goal_flag and loop < max_loop:
                # Move target agents if possible
                for i_tgt_ag in [
                    i for i in range(self.task_len) if i not in self.goal_ag_idx_lst
                ]:
                    tgt_ag_next_pos = self.tgt_ag_path_lst[i_tgt_ag][
                        self.tgt_cur_steps[i_tgt_ag] + 1
                    ]
                    if (
                        tgt_ag_next_pos not in ag_next_pos_lst
                        and tgt_ag_next_pos not in ag_pre_pos_lst
                    ):
                        ag_pre_pos_lst.append(copy.deepcopy(ag_next_pos_lst[i_tgt_ag]))
                        ag_next_pos_lst[i_tgt_ag] = copy.deepcopy(tgt_ag_next_pos)
                        one_tgt_ag_goal_flag = False

                # Move null agents by one step if possible
                for i_null_ag in range(len(null_ags_path_lst)):
                    if len(null_ags_path_lst[i_null_ag]) > 1:
                        # null_ag_pos_idx_lst_on_null_ag_path: 
                        #   indices where the null agent can be on its path.
                        #   Typically 0, but due to Manhattan distance selection,
                        #   it may differ (if it wasn't the nearest one).
                        null_ag_pos_idx_lst_on_null_ag_path = [
                            i
                            for i, pos in enumerate(null_ags_path_lst[i_null_ag])
                            if pos not in self.ag_cur_pos_lst[self.task_len:]
                            and pos not in ag_next_pos_lst[self.task_len:]
                        ]

                        if len(null_ag_pos_idx_lst_on_null_ag_path) > 0:
                            null_ag_cur_pos_idx = max(null_ag_pos_idx_lst_on_null_ag_path)
                            null_ag_cur_pos = null_ags_path_lst[i_null_ag][null_ag_cur_pos_idx]

                            if (
                                null_ag_cur_pos not in ag_next_pos_lst
                                and null_ag_cur_pos_idx < len(null_ags_path_lst[i_null_ag]) - 1
                            ):
                                null_ag_next_pos = null_ags_path_lst[i_null_ag][
                                    null_ag_cur_pos_idx + 1
                                ]

                                # Move a null agent by swapping with the adjacent agent:
                                #   |■|□|  -->  |□|■|
                                if (
                                    null_ag_next_pos in self.ag_cur_pos_lst
                                    and null_ag_next_pos in ag_next_pos_lst
                                ):
                                    ag_next_pos_lst[
                                        ag_next_pos_lst.index(null_ag_next_pos)
                                    ] = copy.deepcopy(null_ag_cur_pos)
                                    null_ags_path_lst[i_null_ag] = null_ags_path_lst[i_null_ag][1:]

                print(ag_next_pos_lst[: self.task_len])
                self.tgt_ag_path_lst_plot.append(copy.deepcopy(self.tgt_ag_path_lst))
                all_path_lst.append(copy.deepcopy(ag_next_pos_lst))
                self.ag_cur_pos_lst = copy.deepcopy(ag_next_pos_lst)

                loop += 1

            self._update_goal_ag()
            tgt_ags_cur_pos_lst = self.ag_cur_pos_lst[: self.task_len]
            for i_tgt_ag in [i for i in range(self.task_len) if i not in self.goal_ag_idx_lst]:
                self.tgt_cur_steps[i_tgt_ag] = (
                    len(self.tgt_ag_path_lst[i_tgt_ag])
                    - self.tgt_ag_path_lst[i_tgt_ag][::-1].index(
                        tgt_ags_cur_pos_lst[i_tgt_ag]
                    )
                    - 1
                )
            self._update_goal_ag()

        return all_path_lst

    def plot_animation(
        self, 
        animation_name: str, 
        routes: list[list[tuple[int, int]]], 
        small_env: bool = False,
        slow_factor: int = 2
    ):
        """
        Draw an animation for the given routes.

        Args:
            animation_name (str): Output file name.
            routes (list[list[tuple[int, int]]]): Sequence of agent positions.
            slow_factor (int): Number of frames per move to interpolate.
        """
        if small_env:
            fig, ax = plt.subplots(figsize=(4.0, 1.25))
        else:
            fig, ax = plt.subplots(figsize=(7, 3.2))

        ax.set_xlim(-0.6, self.size_x - 1 + 0.5)
        ax.set_ylim(-0.55, self.size_y - 1 + 0.55)
        ax.set(xticks=[], yticks=[])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        def update(frame):
            ax.cla()

            tgt_color = ["red", "blue", "green", "steelblue", "deeppink", "brown", "orange",
                         "purple", "olive", "cyan", "lime", "magenta", "yellow", "sandybrown"]

            t = int(frame / slow_factor)
            cur_pos_lst = [routes[t][i] for i in range(len(routes[t]))]
            if t < len(routes) - 1:
                next_pos_lst = [routes[t + 1][i] for i in range(len(routes[t + 1]))]
            else:
                next_pos_lst = cur_pos_lst

            # Interpolate between current and next positions
            w = (frame % slow_factor) / slow_factor
            mid_pos_lst = [
                (
                    (1 - w) * cur_pos_lst[i][0] + w * next_pos_lst[i][0],
                    (1 - w) * cur_pos_lst[i][1] + w * next_pos_lst[i][1],
                )
                for i in range(len(cur_pos_lst))
            ]

            for i in range(len(routes[t])):
                if i < self.task_len:
                    ax.scatter(
                        mid_pos_lst[i][0] - 0.05,
                        mid_pos_lst[i][1],
                        s=56,
                        color=tgt_color[i],
                        label="target agent",
                        marker="o",
                        alpha=0.5,
                    )
                elif i == self.task_len + 1:
                    ax.scatter(
                        mid_pos_lst[i][0] - 0.05,
                        mid_pos_lst[i][1],
                        s=56,
                        color="grey",
                        label="obstructing agent",
                        marker="o",
                        alpha=0.5,
                    )
                else:
                    ax.scatter(
                        mid_pos_lst[i][0] - 0.05,
                        mid_pos_lst[i][1],
                        s=56,
                        color="grey",
                        marker="o",
                        alpha=0.5,
                    )

            for i in range(len(self.static_obss)):
                ax.scatter(
                    self.static_obss[i][0],
                    self.static_obss[i][1],
                    s=56,
                    color="black",
                    marker="s",
                )

            for i_target, tgt_path in enumerate(
                self.tgt_ag_path_lst_plot[min(t, len(self.tgt_ag_path_lst_plot) - 1)]
            ):
                for i in range(len(tgt_path) - 1):
                    ax.plot(
                        [tgt_path[i][0], tgt_path[i + 1][0]],
                        [tgt_path[i][1], tgt_path[i + 1][1]],
                        color=tgt_color[i_target],
                        alpha=0.5,
                    )

            ax.set_xlim(-0.6, self.size_x - 1 + 0.5)
            # Make tick labels smaller than default.
            ax.tick_params(axis="x", labelsize=plt.rcParams["font.size"] / 2)
            ax.set_ylim(-0.55, self.size_y - 1 + 0.55)
            ax.tick_params(axis="y", labelsize=plt.rcParams["font.size"] / 2)

            ax.set_title(f"t: {t}")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

            return ax.plot()

        if small_env:
            plt.subplots_adjust(left=0.005, right=0.52, bottom=0.01, top=0.8)
        else:
            plt.subplots_adjust(left=0.005, right=0.7, bottom=0.01, top=0.92)
        ani = anm.FuncAnimation(
            fig, update, interval=300, frames=len(routes) * slow_factor, repeat=False
        )
        plt.close()
        ani.save(animation_name, fps=8, dpi=300)
