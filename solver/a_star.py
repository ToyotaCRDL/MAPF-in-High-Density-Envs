import heapq
from itertools import count
from math import fabs

import numpy as np

class Location:
    def __init__(self, x: int = -1, y: int = -1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str((self.x, self.y))


class State:
    def __init__(self, time: int, location: Location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self) -> int:
        return hash(str(self.time) + str(self.location.x) + str(self.location.y))

    def is_equal_except_time(self, state) -> bool:
        return self.location == state.location

    def return_location_lst(self) -> list[int, int]:
        return [self.location.x, self.location.y]

    def __str__(self) -> str:
        return str((self.time, self.location.x, self.location.y))


class AStarFollowingConflict():
    def __init__(
        self,
        dimension: tuple[int, int],
        agent: dict,
        obstacles: set,
        moving_obstacles: list[tuple[int, int, int]] = None,
        moving_obstacle_edges: list[tuple[int, int, int, int]] = None,
        a_star_max_iter: int = -1,
        agent_start_pos_lst: list[tuple[int, int]] = None,
        null_agent_pos_lst: list[tuple[int, int]] = None,
        is_dst_add: bool = True,
        considering_cycle_conflict: bool = True,
    ):
        if moving_obstacles is None:
            moving_obstacles = []
        if moving_obstacle_edges is None:
            moving_obstacle_edges = []
        if agent_start_pos_lst is None:
            agent_start_pos_lst = []
        if null_agent_pos_lst is None:
            null_agent_pos_lst = []

        self.dimension = dimension
        self.obstacles = obstacles
        self.moving_obstacles = moving_obstacles
        self.moving_obstacle_edges = moving_obstacle_edges
        self.a_star_max_iter = a_star_max_iter
        self.agent_start_pos_lst = agent_start_pos_lst
        self.null_agent_pos_lst = null_agent_pos_lst
        self.is_dst_add = is_dst_add
        self.considering_cycle_conflict = considering_cycle_conflict

        start_state = State(0, Location(agent["start"][0], agent["start"][1]))
        goal_state = State(0, Location(agent["goal"][0], agent["goal"][1]))
        self.agent = {"start": start_state, "goal": goal_state}
        
        self.iter = 0

    def _get_neighbors(self, state: State) -> list[State]:
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y + 1))
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y - 1))
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x - 1, state.location.y))
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x + 1, state.location.y))
        if self._state_valid(n) and self._transition_valid(state, n):
            neighbors.append(n)
        return neighbors

    def _get_all_obstacles(self, time: int) -> set[tuple[int, int]]:
        all_obs = set()
        for o in self.moving_obstacles:
            if o[2] < 0 and time >= -o[2]:
                all_obs.add((o[0], o[1]))
        return self.obstacles | all_obs

    def _state_valid(self, state: State) -> bool:
        return (
            state.location.x >= 0
            and state.location.x < self.dimension[0]
            and state.location.y >= 0
            and state.location.y < self.dimension[1]
            and (state.location.x, state.location.y)
            not in self._get_all_obstacles(state.time)
        )

    def _find_cycles(self, idx_vectors: list[tuple[int, int]]) -> bool:
        edge_map = {start: end for start, end in idx_vectors}

        # Check cycle
        for start in edge_map:
            visited = {start}
            current = start
            while True:
                if edge_map[current] in visited:
                    return True
                if edge_map[current] not in edge_map:
                    break
                visited.add(edge_map[current])
                current = edge_map[current]

        return False

    def _transition_valid(self, state_cur: State, state_next: State) -> bool:
        # Vertex conflicts against moving obstacles
        if (state_next.location.x, state_next.location.y, state_next.time) in (
            self.moving_obstacles
        ):
            return False
        # Following conflicts against moving obstacles (including edge conflicts)
        if (state_next.location.x, state_next.location.y, state_next.time - 1) in (
            self.moving_obstacles
        ):
            return False
        if (state_next.location.x, state_next.location.y, state_next.time + 1) in (
            self.moving_obstacles
        ):
            return False
        # Edge conflicts against moving obstacle (without considering time)
        if (
            state_next.location.x,
            state_next.location.y,
            state_cur.location.x,
            state_cur.location.y,
        ) in self.moving_obstacle_edges:
            return False
        # Cycle conflicts against moving obstacle (without considering time)
        if self.considering_cycle_conflict and len(self.moving_obstacle_edges) > 0:
            cur_pos_lst = [(state_cur.location.x, state_cur.location.y)] + [
                (mo[0], mo[1]) for mo in self.moving_obstacle_edges
            ]
            next_pos_lst = [(state_next.location.x, state_next.location.y)] + [
                (mo[2], mo[3]) for mo in self.moving_obstacle_edges
            ]

            dup_idx_pairs = []
            for i_cur_pos, tgt_ag_cur_pos in enumerate(cur_pos_lst):
                for i_next_pos, tgt_ag_next_pos in enumerate(next_pos_lst):
                    if i_cur_pos != i_next_pos and tgt_ag_cur_pos == tgt_ag_next_pos:
                        dup_idx_pairs.append((i_cur_pos, i_next_pos))

            if self._find_cycles(dup_idx_pairs):
                return False

        return True

    def _admissible_heuristic(self, state: State) -> float:
        goal = self.agent["goal"]
        return fabs(state.location.x - goal.location.x) + fabs(
            state.location.y - goal.location.y
        )

    def _is_at_goal(self, state: State) -> bool:
        goal_state = self.agent["goal"]
        return state.is_equal_except_time(goal_state)

    def _reconstruct_path(self, came_from: dict, current: State) -> list[State]:
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def _search(self) -> list[State]:
        initial_state = self.agent["start"]
        step_cost = 1

        closed_set = set()
        open_set = {initial_state}

        came_from = {}

        g_score = {initial_state: 0}

        h_score = self._admissible_heuristic(initial_state)
        f_score = {initial_state: h_score}

        heap = []
        index = count(0)
        heapq.heappush(
            heap, (f_score[initial_state], h_score, next(index), initial_state)
        )

        while open_set and (self.a_star_max_iter == -1 or self.iter < self.a_star_max_iter):
            self.iter = self.iter + 1
            if self.iter == self.a_star_max_iter:
                pass

            current = heapq.heappop(heap)[3]

            if self._is_at_goal(current):
                return self._reconstruct_path(came_from, current)

            open_set -= {current}
            closed_set |= {current}

            neighbor_list = self._get_neighbors(current)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                dst_add = 0
                if self.is_dst_add:
                    nx, ny = neighbor.location.x, neighbor.location.y
                    neighbor_xy = np.array((nx, ny))
                    null_xy = np.array(self.null_agent_pos_lst)
                    dst_to_null = np.abs(null_xy - neighbor_xy).sum(axis=1).min()
                    if dst_to_null > 0 and dst_to_null > tentative_g_score:
                        dst_add = dst_to_null - tentative_g_score

                if neighbor not in open_set:
                    open_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                h_score = self._admissible_heuristic(neighbor)
                f_score[neighbor] = g_score[neighbor] + h_score + dst_add
                heapq.heappush(heap, (f_score[neighbor], h_score, next(index), neighbor))
        return False

    def compute_solution(self) -> list[dict]:
        local_solution = self._search()
        if not local_solution:
            return {}
        
        path_dict_list = [
            {"t": state.time, "x": state.location.x, "y": state.location.y}
            for state in local_solution
        ]
        
        return path_dict_list
