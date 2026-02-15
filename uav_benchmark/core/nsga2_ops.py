from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from uav_benchmark.core.chromosome import Chromosome


def n_d_sort(pop_obj: np.ndarray, pop_con: np.ndarray | None, n_sort: int) -> tuple[np.ndarray, int]:
    pop_obj = np.asarray(pop_obj, dtype=float)
    population_size, objective_count = pop_obj.shape
    if pop_con is not None:
        pop_con = np.asarray(pop_con, dtype=float)
        infeasible = np.any(pop_con > 0, axis=1)
        if np.any(infeasible):
            max_values = np.max(pop_obj, axis=0, keepdims=True)
            penalties = np.sum(np.maximum(0.0, pop_con[infeasible]), axis=1, keepdims=True)
            pop_obj[infeasible] = max_values + penalties
    if objective_count < 5 or population_size < 500:
        return _ens_ss(pop_obj, n_sort)
    return _t_ens(pop_obj, n_sort)


def _ens_ss(pop_obj: np.ndarray, n_sort: int) -> tuple[np.ndarray, int]:
    unique_obj, inverse = np.unique(pop_obj, axis=0, return_inverse=True)
    table = np.bincount(inverse, minlength=unique_obj.shape[0])
    n_unique, objective_count = unique_obj.shape
    front_no = np.full(n_unique, np.inf, dtype=float)
    max_front = 0
    while np.sum(table[np.isfinite(front_no)]) < min(n_sort, inverse.shape[0]):
        max_front += 1
        for candidate_index in range(n_unique):
            if np.isfinite(front_no[candidate_index]):
                continue
            dominated = False
            for compare_index in range(candidate_index - 1, -1, -1):
                if front_no[compare_index] != max_front:
                    continue
                compare_objective = 1
                while compare_objective < objective_count and unique_obj[candidate_index, compare_objective] >= unique_obj[compare_index, compare_objective]:
                    compare_objective += 1
                dominated = compare_objective >= objective_count
                if dominated or objective_count == 2:
                    break
            if not dominated:
                front_no[candidate_index] = max_front
    return front_no[inverse], max_front


def _t_ens(pop_obj: np.ndarray, n_sort: int) -> tuple[np.ndarray, int]:
    unique_obj, inverse = np.unique(pop_obj, axis=0, return_inverse=True)
    table = np.bincount(inverse, minlength=unique_obj.shape[0])
    n_unique, objective_count = unique_obj.shape
    front_no = np.full(n_unique, np.inf, dtype=float)
    max_front = 0
    forest = np.zeros(n_unique, dtype=int)
    children = np.zeros((n_unique, objective_count - 1), dtype=int)
    left_child = np.full(n_unique, objective_count, dtype=int)
    father = np.zeros(n_unique, dtype=int)
    brother = np.full(n_unique, objective_count, dtype=int)
    objective_order = np.argsort(-unique_obj[:, 1:], axis=1) + 1
    while np.sum(table[np.isfinite(front_no)]) < min(n_sort, inverse.shape[0]):
        max_front += 1
        root = int(np.where(~np.isfinite(front_no))[0][0])
        forest[max_front - 1] = root
        front_no[root] = max_front
        for point_index in range(n_unique):
            if np.isfinite(front_no[point_index]):
                continue
            pruning = np.zeros(n_unique, dtype=int)
            query = forest[max_front - 1]
            while True:
                objective_cursor = 0
                while objective_cursor < objective_count - 1 and unique_obj[point_index, objective_order[query, objective_cursor]] >= unique_obj[query, objective_order[query, objective_cursor]]:
                    objective_cursor += 1
                if objective_cursor == objective_count - 1:
                    break
                pruning[query] = objective_cursor + 1
                if left_child[query] <= pruning[query]:
                    query = children[query, left_child[query] - 1]
                else:
                    while father[query] != 0 and brother[query] > pruning[father[query]]:
                        query = father[query]
                    if father[query] != 0:
                        query = children[father[query], brother[query] - 1]
                    else:
                        break
            if objective_cursor < objective_count - 1:
                front_no[point_index] = max_front
                query = forest[max_front - 1]
                while children[query, pruning[query] - 1] != 0:
                    query = children[query, pruning[query] - 1]
                children[query, pruning[query] - 1] = point_index
                father[point_index] = query
                if left_child[query] > pruning[query]:
                    brother[point_index] = left_child[query]
                    left_child[query] = pruning[query]
                else:
                    sibling = children[query, left_child[query] - 1]
                    while brother[sibling] < pruning[query]:
                        sibling = children[query, brother[sibling] - 1]
                    brother[point_index] = brother[sibling]
                    brother[sibling] = pruning[query]
    return front_no[inverse], max_front


def crowding_distance(pop_obj: np.ndarray, front_no: np.ndarray) -> np.ndarray:
    pop_obj = np.asarray(pop_obj, dtype=float)
    front_no = np.asarray(front_no, dtype=float)
    n_points, n_objectives = pop_obj.shape
    distance = np.zeros(n_points, dtype=float)
    fronts = [item for item in np.unique(front_no) if np.isfinite(item)]
    for front in fronts:
        members = np.where(front_no == front)[0]
        if members.size == 0:
            continue
        finite_members = members[np.all(np.isfinite(pop_obj[members]), axis=1)]
        if finite_members.size == 0:
            continue
        objective_max = np.max(pop_obj[finite_members], axis=0)
        objective_min = np.min(pop_obj[finite_members], axis=0)
        for objective_index in range(n_objectives):
            order = finite_members[np.argsort(pop_obj[finite_members, objective_index])]
            if order.size == 0:
                continue
            distance[order[0]] = np.inf
            distance[order[-1]] = np.inf
            denominator = objective_max[objective_index] - objective_min[objective_index]
            if not np.isfinite(denominator) or denominator <= 0:
                continue
            for rank in range(1, order.size - 1):
                distance[order[rank]] += (pop_obj[order[rank + 1], objective_index] - pop_obj[order[rank - 1], objective_index]) / denominator
    return distance


def tournament_selection(k_tournament: int, n_select: int, *fitness_values: np.ndarray) -> np.ndarray:
    reshaped = [np.asarray(value).reshape(-1, 1) for value in fitness_values]
    merged = np.hstack(reshaped)
    rank = np.argsort(np.lexsort(np.flipud(merged.T)))
    rank_values = np.empty_like(rank)
    rank_values[rank] = np.arange(rank.size)
    candidates = np.random.randint(0, merged.shape[0], size=(k_tournament, n_select))
    best_rows = np.argmin(rank_values[candidates], axis=0)
    columns = np.arange(n_select)
    return candidates[best_rows, columns]


def environmental_selection(
    population: Sequence[Chromosome],
    n_keep: int,
    objective_count: int,
    use_constraints: bool = False,
) -> tuple[list[Chromosome], np.ndarray, np.ndarray]:
    objective_matrix = np.array([individual.objs for individual in population], dtype=float)
    constraint_matrix = None
    if use_constraints:
        constraint_matrix = np.array([[individual.cons] for individual in population], dtype=float)
    front_no, max_front = n_d_sort(objective_matrix, constraint_matrix, n_keep)
    next_mask = front_no < max_front
    crowding = crowding_distance(objective_matrix, front_no)
    last_members = np.where(front_no == max_front)[0]
    if np.sum(next_mask) < n_keep and last_members.size > 0:
        order = last_members[np.argsort(-crowding[last_members])]
        required = n_keep - int(np.sum(next_mask))
        next_mask[order[:required]] = True
    selected_population = [population[index] for index in np.where(next_mask)[0]]
    return selected_population, front_no[next_mask], crowding[next_mask]


def f_operator(population: Sequence[Chromosome], mating_pool: np.ndarray, boundary: np.ndarray, model: dict) -> list[Chromosome]:
    n_select = int(mating_pool.shape[0])
    if n_select == 0:
        return []
    if n_select % 2 == 1:
        # Mirror MATLAB robustness fix for odd mating pools:
        # duplicate one parent to make pairwise crossover valid, then trim.
        extra_parent = int(mating_pool[np.random.randint(0, n_select)])
        mating_pool = np.concatenate([mating_pool, np.array([extra_parent], dtype=mating_pool.dtype)])
    point_count = population[0].rnvec.shape[0]
    parent_tensor = np.stack([population[index].rnvec for index in mating_pool], axis=0)
    even_count = int(mating_pool.shape[0])
    half = even_count // 2
    parent_one = parent_tensor[:half].reshape(-1, 3)
    parent_two = parent_tensor[half : 2 * half].reshape(-1, 3)

    pro_c = 1.0
    pro_m = 1.0 / 3.0
    dis_c = 20.0
    dis_m = 20.0
    beta = np.zeros_like(parent_one)
    random_mu = np.random.rand(*beta.shape)
    mask = random_mu <= 0.5
    beta[mask] = (2.0 * random_mu[mask]) ** (1.0 / (dis_c + 1.0))
    beta[~mask] = (2.0 - 2.0 * random_mu[~mask]) ** (-1.0 / (dis_c + 1.0))
    beta *= (-1) ** np.random.randint(0, 2, size=beta.shape)
    beta[np.random.rand(*beta.shape) < 0.5] = 1.0
    row_mask = np.random.rand(beta.shape[0], 1) > pro_c
    beta[row_mask.repeat(beta.shape[1], axis=1)] = 1.0

    offspring = np.vstack(
        [
            (parent_one + parent_two) / 2.0 + beta * (parent_one - parent_two) / 2.0,
            (parent_one + parent_two) / 2.0 - beta * (parent_one - parent_two) / 2.0,
        ]
    )
    max_values = np.repeat(boundary[0:1], offspring.shape[0], axis=0)
    min_values = np.repeat(boundary[1:2], offspring.shape[0], axis=0)
    mutation_mask = np.random.rand(*offspring.shape) <= pro_m
    mutation_mu = np.random.rand(*offspring.shape)
    lower_mask = mutation_mask & (mutation_mu < 0.5)
    denominator = max_values - min_values
    offspring[lower_mask] += denominator[lower_mask] * (
        (2.0 * mutation_mu[lower_mask] + (1.0 - 2.0 * mutation_mu[lower_mask]) * (1.0 - (offspring[lower_mask] - min_values[lower_mask]) / denominator[lower_mask]) ** (dis_m + 1.0))
        ** (1.0 / (dis_m + 1.0))
        - 1.0
    )
    upper_mask = mutation_mask & (mutation_mu >= 0.5)
    offspring[upper_mask] += denominator[upper_mask] * (
        1.0
        - (
            2.0 * (1.0 - mutation_mu[upper_mask])
            + 2.0
            * (mutation_mu[upper_mask] - 0.5)
            * (1.0 - (max_values[upper_mask] - offspring[upper_mask]) / denominator[upper_mask]) ** (dis_m + 1.0)
        )
        ** (1.0 / (dis_m + 1.0))
    )
    offspring = np.clip(offspring, min_values, max_values)

    children: list[Chromosome] = []
    for child_index in range(n_select):
        child = population[mating_pool[child_index]].copy()
        start = child_index * point_count
        end = (child_index + 1) * point_count
        child.rnvec = offspring[start:end]
        order = np.argsort(child.rnvec[:, 0], kind="mergesort")
        child.rnvec = child.rnvec[order]
        child.path = child.rnvec.copy()
        child.adjust_constraint_turning_angle(model)
        child.evaluate(model)
        children.append(child)
    return children
