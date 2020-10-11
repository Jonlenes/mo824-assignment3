import sys
import itertools

import pandas as pd
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB, tuplelist, quicksum

from read_instance import load_instance, list_avaliable_instances


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


def get_optimal_tour(model, num_vertices):
    """
    Função executada para gerar lista com a sequencia das cidades visitadas
    """
    tours, edges = [], []
    for var in model._edges_variables:
        # Obtendo arestas visitadas
        values = model.getAttr("x", var)
        selected = gp.tuplelist((i, j) for i, j in values.keys() if values[i, j] > 0.5)
        edges.append(list(selected))

        # Obtendo caminho percorrido
        tour = get_smallest_subtour(num_vertices, selected)
        tours.append(tour)

    # Check if the solution is valid
    check_2tsp_valid_solution(num_vertices, tours, edges)

    return tours, edges


def check_2tsp_valid_solution(num_vertices, tours, edges):
    """
    Verfica se solução fornecida é valida
    """
    # Verifica se todos os vertices foram utilizados
    assert not any([len(tour) != num_vertices for tour in tours])
        
    # Verifica a dijunção entre os dois ocnjuntos de arestas
    assert not any([edge in edges[1] for edge in edges[0]])


def build_2tsp_model(num_vertices, points, dist):
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
        gp.quicksum(x_edges[i, j] * dist[i, j] for i, j in dist.keys())
        + gp.quicksum(y_edges[i, j] * dist[i, j] for i, j in dist.keys()),
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


def main(ins_folder):
    """
    Run 2TSP model in all instances and save an CSV with the results
    """
    ins_filenames = list_avaliable_instances(ins_folder)
    results = pd.DataFrame()

    print("Starting experiments")
    for filename in tqdm(ins_filenames):
        # Load instance
        instance = load_instance(filename)

        # Save cost and time
        costs, times, optimal_tours, selected_edges = [], [], [], []

        # Building the models
        for model, callback in [build_2tsp_model(*instance)]:
            # Optimize model
            model.optimize(callback)
            # Saving costs and times
            costs.append(model.objVal)
            times.append(model.runtime)

            # Get optimal tour
            tours, edges = get_optimal_tour(model, instance[0])
            optimal_tours.append(tours)
            selected_edges.append(edges)

        results = results.append(
            {
                "n_vertices": instance[0],

                "lim_inf_lag": 0,
                "lim_sup_lag": 0,
                "time_lag": round(times[0], 3),
                "cost_lag": costs[0],
                "optimal_tour_lag": optimal_tours[0],
                "edges_lag": selected_edges[0],

                "lim_inf_int": 0,
                "lim_sup_int": 0,
                "time_int": round(times[0], 3),
                "cost_int": costs[0],
                "optimal_tour_int": optimal_tours[0],
                "edges_int": selected_edges[0]
            },
            ignore_index=True,
        )

    results.to_csv("data/results.csv", index=False)



if __name__ == "__main__":
    ins_folder = "data"
    if len(sys.argv) > 1:
        ins_folder = sys.argv[1]
    main(ins_folder)
