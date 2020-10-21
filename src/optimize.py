import sys
import itertools
import numpy as np

import pandas as pd
import gurobipy as gp
from gurobipy import GRB, tuplelist, quicksum
from tqdm import tqdm

from read_instance import load_instance, list_avaliable_instances
from subgradient import generic_subgradient


def eliminate_subtour_solution(num_vertices, model, where):
    """
    # Função executada pelo gurobi ao avaliar um candidato, esperamos
    # elimitar soluções compostas por subciclos
    """
    # Candidato a solução, segundo exemplo do Gurobi
    if where == GRB.Callback.MIPSOL:
        # Validando os caminhos percorridos por cada variavel
        for v in model._edges_variables:
            values = model.cbGetSolution(v)
            selected = gp.tuplelist((i, j) for i, j in v.keys() if values[i, j] > 0.5)

            # Valiando subtours percorridos
            tour = get_smallest_subtour(num_vertices, selected)

            # Verificando se o candidato respeita as condições
            if len(tour) < num_vertices:
                # Adiciona constraint para impedir a utilização desse subtour candidato
                model.cbLazy(
                    quicksum(v[i, j] for i, j in itertools.combinations(tour, 2))
                    <= len(tour) - 1
                )


def get_smallest_subtour(num_vertices, edges):
    """
    Encontra todos os ciclos percorridos pelos vertices
    """
    unvisited = list(range(num_vertices))
    cycle = range(num_vertices + 1)  # initial length has 1 more city
    while unvisited:
        tour = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            tour.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, "*") if j in unvisited]
        if len(cycle) > len(tour):
            cycle = tour
    return cycle


def get_optimal_tour(var_edges, num_vertices):
    """
    Função executada para gerar lista com a sequencia das cidades visitadas
    """
    tours, edges = [], []
    for var in var_edges:
        selected = gp.tuplelist((i, j) for i, j in var.keys() if var[i, j] > 0.5)
        edges.append(list(selected))

        # Obtendo caminho percorrido
        tour = get_smallest_subtour(num_vertices, selected)
        tours.append(tour)
    return tours, edges


def check_2tsp_valid_solution(num_vertices, tours, edges):
    """
    Verfica se solução fornecida é valida
    """
    # Verifica se todos os vertices foram utilizados
    assert not any([len(tour) != num_vertices for tour in tours])
    # Verifica a dijunção entre os dois ocnjuntos de arestas
    assert not any([edge in edges[1] for edge in edges[0]])


def try_swap_edges(x, y, num_vertices, edges, u1, v1, u2, v2):
    # Verifica se as arestas estão disponiveis
    if x[u1, v2] + x[u2, v1] + y[u1, v2] + y[u2, v1] > 0:
        return False, None
    # Realiza a troca das arestas
    used_edges = x.copy()
    used_edges[u1, v1] = used_edges[v1, u1] = 0
    used_edges[u2, v2] = used_edges[v2, u2] = 0
    used_edges[u1, v2] = used_edges[v2, u1] = 1
    used_edges[u2, v1] = used_edges[v1, u2] = 1

    # Verifica se a troca é válida
    selected = gp.tuplelist((i, j) for i, j in edges if used_edges[i, j] > 0.5)
    valid = len(get_smallest_subtour(num_vertices, selected)) == num_vertices

    return valid, used_edges


def build_2tsp_model(num_vertices, dist):
    """
    Build the 2tsp model
    """
    # Create a new model
    model = gp.Model("2tsp")

    # Set Params
    model.setParam(gp.GRB.Param.OutputFlag, 0)
    model.setParam(gp.GRB.Param.TimeLimit, 30 * 60)
    model.setParam(gp.GRB.Param.Seed, 42)

    # Variables
    x_edges = model.addVars(dist.keys(), vtype=GRB.BINARY, name="x_edges")
    y_edges = model.addVars(dist.keys(), vtype=GRB.BINARY, name="y_edges")
    for i, j in x_edges.keys():
        x_edges[j, i] = x_edges[i, j]  # edge in opposite direction
        y_edges[j, i] = y_edges[i, j]  # edge in opposite direction
    # Objective
    model.setObjective(
        gp.quicksum(
            x_edges[i, j] * dist[i, j] + y_edges[i, j] * dist[i, j]
            for i, j in dist.keys()
        ),
        GRB.MINIMIZE,
    )

    # Constraints
    #   Add degree-2 constraint
    model.addConstrs(x_edges.sum(i, "*") == 2 for i in range(num_vertices))
    model.addConstrs(y_edges.sum(i, "*") == 2 for i in range(num_vertices))

    model.addConstrs(
        gp.quicksum([x_edges[i, j], y_edges[i, j]]) <= 1 for i, j in x_edges.keys()
    )

    model._edges_variables = [x_edges, y_edges]
    model.Params.lazyConstraints = 1

    return model, (
        lambda model, where: eliminate_subtour_solution(num_vertices, model, where)
    )


def lag_heuristic_2tps(model, num_vertices, dist):
    """
    Lagrangian heuristic: takes the LLB2TPS solution and transforms it into a valid 2TPS
    solution by swapping the edges. This will be a upper bound for 2TPS
    """
    edges_variables = model._edges_variables
    x = model.getAttr("X", edges_variables[0])
    y = model.getAttr("X", edges_variables[1])
    edges = edges_variables[0].keys()

    for u1, v1 in edges:
        # Verifica se a restrição foi violada
        if x[u1, v1] + y[u1, v1] <= 1:
            continue
        # Busca outra aresta que pode ser trocada com essa
        for u2, v2 in edges:
            # Verifica se tem algum vertice em comum
            if len(set([u1, v1, u2, v2])) != 4:
                continue
            # Tenta fazer a troca de aresta, verificando se nenhum subciclo foi gerado
            success, used_edges = try_swap_edges(x, y, num_vertices, edges, u1, v1, u2, v2)
            if success:
                x = used_edges
                break

    # Sanity check: verifica se a heurística encontrou uma solução valida
    tours, edges = get_optimal_tour([x, y], num_vertices)
    check_2tsp_valid_solution(num_vertices, tours, edges)

    # Computa o custo da solução heurística
    cost = sum((x[i, j] * dist[i, j] + y[i, j] * dist[i, j]) for i, j in dist.keys())

    return cost


def solve_llb2tsp(model, dist, num_vertices, u):
    """
    Given a vector, u (Lagrange multipliers), solve the LLB2TPS and return the value
    of the objective function (which will be the lower bound for the for 2TPS)
    """
    x_edges = model._edges_variables[0]
    y_edges = model._edges_variables[1]

    # Set objective
    model.setObjective(
        gp.quicksum(
            x_edges[i, j] * dist[i, j] + y_edges[i, j] * dist[i, j]
            for i, j in dist.keys()
        )
        + gp.quicksum(
            u[index] * (-1 + (x_edges[i, j] + y_edges[i, j]))
            for index, (i, j) in enumerate(x_edges.keys())
        ),
        GRB.MINIMIZE,
    )

    model.optimize(
        lambda model, where: eliminate_subtour_solution(num_vertices, model, where)
    )

    return model.ObjVal


def compute_subgradient(model):
    """
    Computes the subgradients considering the current solution provided by LLB2TPS.
    This is done using the dualized constraint as follows: b - Ax.
    """
    edges_variables = model._edges_variables
    x_edges_values = model.getAttr("X", edges_variables[0])
    y_edges_values = model.getAttr("X", edges_variables[1])
    return np.array(
        [
            -1 + (x_edges_values[i, j] + y_edges_values[i, j])
            for i, j in model._edges_variables[0].keys()
        ]
    )


def build_llb2tsp_model(num_vertices, dist):
    """
    Build the llb2tsp model
    """
    # Create a new model
    model = gp.Model("llb2tsp")

    # Set Params
    model.setParam(gp.GRB.Param.OutputFlag, 0)
    model.setParam(gp.GRB.Param.TimeLimit, 30 * 60)
    model.setParam(gp.GRB.Param.Seed, 42)

    # Variables
    x_edges = model.addVars(dist.keys(), vtype=GRB.BINARY, name="x_edges")
    y_edges = model.addVars(dist.keys(), vtype=GRB.BINARY, name="y_edges")
    for i, j in x_edges.keys():
        x_edges[j, i] = x_edges[i, j]  # edge in opposite direction
        y_edges[j, i] = y_edges[i, j]  # edge in opposite direction
    
    # Constraints
    #   Add degree-2 constraint
    model.addConstrs(x_edges.sum(i, "*") == 2 for i in range(num_vertices))
    model.addConstrs(y_edges.sum(i, "*") == 2 for i in range(num_vertices))

    model._edges_variables = [x_edges, y_edges]
    model.Params.lazyConstraints = 1

    return model


def main(ins_folder):
    """
    Run LLB2TPS and 2TSP model in one instance and save an CSV with the results
    """
    # If True, will run the subgradient method for the LLB2TPS
    run_llb2tps = True
    # If True, will optimize the 2TSP (with ILP)
    run_2tps_ilp = True
    # Choose: {0, 1, 2, 3, 4}
    instance_id = 0

    # Load instance
    num_vertices, _, dist = load_instance(f"{ins_folder}/instancia-{instance_id}.json")
    print(f"Running instance {instance_id} with {num_vertices} vertices")

    # Begin: LLB2TPS
    if run_llb2tps:
        # Building the model
        model = build_llb2tsp_model(num_vertices, dist)

        func_Z_lb = lambda u: solve_llb2tsp(model, dist, num_vertices, u)
        func_Z_ub = lambda : lag_heuristic_2tps(model, num_vertices, dist)
        func_csg = lambda : compute_subgradient(model)

        # Run subgradient method
        results_llb2tps = generic_subgradient(
            model,
            func_Z_lb=func_Z_lb,
            func_Z_ub=func_Z_ub,
            func_compute_subgradient=func_csg,
            func_pi=lambda k: (0.999 ** k) * 2,
            u=[0] * int(num_vertices * (num_vertices - 1)),
            n_iter=100,
            verbose=True,
        )
    else:
        results_llb2tps = {"Z_lb": 0, "Z_ub": 0, "time": 0}
    # End: LLB2TPS

    # Begin: 2TPS with ILP
    if run_2tps_ilp:
        # Building the model
        model, callback = build_2tsp_model(num_vertices, dist)

        # Optimaze the model
        model.optimize(callback)

        results_2tps = {
            "Z_lb": model.ObjBound,
            "Z_ub": model.objVal,
            "time": model.runtime,
        }
    else:
        results_2tps = {"Z_lb": 0, "Z_ub": 0, "time": 0}
    # End: 2TPS with ILP

    # Save csv with the results
    pd.DataFrame(
        {
            "n_vertices": num_vertices,
            "Z_lb_lag": round(results_llb2tps["Z_lb"], 2),
            "Z_ub__lag": round(results_llb2tps["Z_ub"], 2),
            "time_lag": round(results_llb2tps["time"], 2),
            "Z_lb_lip": round(results_2tps["Z_lb"], 2),
            "Z_ub_lip": round(results_2tps["Z_ub"], 2),
            "time_lip": round(results_2tps["time"], 2),
        },
        index=[0],
    ).to_csv(f"data/results_{instance_id}.csv", index=False)


if __name__ == "__main__":
    ins_folder = "data"
    if len(sys.argv) > 1:
        ins_folder = sys.argv[1]
    main(ins_folder)
