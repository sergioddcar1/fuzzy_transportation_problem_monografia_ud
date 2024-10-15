import pulp
import numpy as np
from pprint import pprint
import sys
import random
import matplotlib.pyplot as plt
import networkx as nx

# Clase para manejar números triangulares fuzzy
class TriangularFuzzyNumber:
    
    def parametric_form(self, tuple_tfn):
        lower = tuple_tfn[0]
        middle = tuple_tfn[1]
        upper = tuple_tfn[2]
        lower_param = [lower + (middle - lower) * 0, lower + (middle - lower) * 1]
        upper_param = [upper - (upper - middle) * 0, upper - (upper - middle) * 1]
        return lower_param + upper_param
    
    def sum_triangular_fuzzy_number(self, tuple1, tuple2):
        tuple3 = (tuple1[0]+tuple2[0], tuple1[1]+tuple2[1], tuple1[2]+tuple2[2])
        return tuple3
    
    def rest_triangular_fuzzy_number(tuple1, tuple2):
        tuple3 = (tuple1[0]-tuple2[2], tuple1[1]-tuple2[1], tuple1[2]-tuple2[0])
        return tuple3

    def constant_multiplication(self, p, tuple1):
        tuple2 = (tuple1[0]*p, tuple1[1]*p, tuple1[2]*p)
        return tuple2
    def generate_triangular_fuzzy_number(self, lower_limit=0, upper_limit=100):

        # Step 1: Generate random numbers using Gaussian distribution for smoother randomness
        base_numbers = [random.gauss((upper_limit + lower_limit) / 2, (upper_limit - lower_limit) / 6) for _ in range(3)]
        
        # Step 2: Sort the numbers to satisfy the trapezoidal condition
        base_numbers.sort()
        
        # Step 3: Apply constraints to adjust differences between them (optional)
        min_distance = round((upper_limit - lower_limit) * 0.1, 0)  # Ensure a minimum distance between points
        
        for i in range(1, 3):
            if base_numbers[i] - base_numbers[i - 1] < min_distance:
                base_numbers[i] = base_numbers[i - 1] + min_distance
        
        # Step 4: Ensure the numbers are within the bounds [a_min, a_max]
        base_numbers = [int(max(min(x, upper_limit), lower_limit)) for x in base_numbers]

        

        return tuple(base_numbers)

# Clase para manejar el problema de optimización usando PuLP
class FuzzyTransportProblem:
    def __init__(self, costs_stage_one, costs_stage_two, supply, demand, z):
        self.costs_stage_one = costs_stage_one  
        self.costs_stage_two = costs_stage_two  
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.z = z
        self.sources = len(self.costs_stage_one[0])
        self.destinations = len(self.costs_stage_two[0][0])
        self.products = len(self.costs_stage_one)
        self.transhipments = len(self.costs_stage_two[0])
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.tfn = TriangularFuzzyNumber()
    
    def create_variables(self):
        self.x = pulp.LpVariable.dicts("x", [(p, i, k) for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)], lowBound=0)
        self.y = pulp.LpVariable.dicts("y", [(p, k, j) for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)], lowBound=0)
    def add_constraints(self):
        # Restricciones de oferta
        for p in range(self.products):
            for i in range(self.sources):
                
                supply_val = self.tfn.parametric_form(self.supply[p][i])
                # pprint(supply_val)
                self.problem += pulp.lpSum([self.x[p, i, k] for k in range(self.transhipments)]) == supply_val[self.z]
            
        # Restricciones de demanda
        for p in range(self.products):
            for j in range(self.destinations):
            
                demand_val = self.tfn.parametric_form(self.demand[p][j])
                pprint(demand_val)
                self.problem += pulp.lpSum([self.y[p, k, j] for k in range(self.transhipments)]) == demand_val[self.z]

        for p in range(self.products):
            for k in range(self.transhipments):

                self.problem += pulp.lpSum([self.x[p, i, k] for i in range(self.sources)]) == pulp.lpSum([self.y[p, k, j] for j in range(self.destinations)])


    def set_objective(self):
        self.problem += pulp.lpSum(self.costs_stage_one[p][i][k] * self.x[p, i, k] for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)) + pulp.lpSum(self.costs_stage_two[p][k][j] * self.y[p, k, j] for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations))
    
    def solve(self):
        self.problem.solve()
        
        return self.problem.status

    def get_solution(self):


        solution_stage_one = {(p, i, k): self.x[p, i, k].varValue for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)}
        solution_stage_two = {(p, k, j): self.y[p, k, j].varValue for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)}
        
        return [solution_stage_one, solution_stage_two]
    
class FuzzyCharts():
    
    
    def show_chart(self, list_tfn):
        i = 0
        self.list_tfn = list_tfn[0]
        # Configurar matplotlib para usar LaTeX
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')  # Usar Computer Modern, que es la fuente por defecto de LaTeX

        x = np.linspace(min([solution[0] for solution in self.list_tfn]) - 5, max([solution[2] for solution in self.list_tfn]) + 5, 500)

        list_a = [solution[0] for solution in self.list_tfn]
        list_b = [solution[1] for solution in self.list_tfn]
        list_c = [solution[2] for solution in self.list_tfn]
        # Definir la función de pertenencia trapezoidal
        

        # Calcular los grados de pertenencia
        y = np.maximum(np.minimum(np.minimum((x-list_a[0])/(list_b[0]-list_a[0]), 1), (list_c[0]-x)/(list_c[0]-list_b[0])), 0)
        z = np.maximum(np.minimum(np.minimum((x-list_a[1])/(list_b[1]-list_a[1]), 1), (list_c[1]-x)/(list_c[1]-list_b[1])), 0)
        w = np.maximum(np.minimum(np.minimum((x-list_a[2])/(list_b[2]-list_a[2]), 1), (list_c[2]-x)/(list_c[2]-list_b[2])), 0)
        # Graficar
        plt.plot(x, y, label=f'Producto {i+1} - Origen 1 ({list_a[0]}, {list_b[0]}, {list_c[0]})')
        plt.plot(x, z, label=f'Producto {i+1} - Origen 2 ({list_a[1]}, {list_b[1]}, {list_c[1]})')
        plt.plot(x, w, label=f'Producto {i+1} - Origen 3 ({list_a[2]}, {list_b[2]}, {list_c[2]})')
        plt.fill_between(x, y, alpha=0.2)
        plt.fill_between(x, z, alpha=0.2)
        plt.fill_between(x, w, alpha=0.2)
        plt.xlabel(r'Unidades monetarias (\$)')
        plt.ylabel(r'Grado de pertenencia $\alpha$')
        plt.title('Número difuso triangular')
        plt.legend()
        plt.grid(True)
        plt.show()

class Network():

    def show_network(self, allocation_stage_one, allocation_stage_two):
        # Crear un grafo vacío
        G = nx.DiGraph()  # Grafo dirigido, porque hay un flujo de origen -> transbordo -> destino
        # Configurar matplotlib para usar LaTeX
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')  # Usar Computer Modern, que es la fuente por defecto de LaTeX

        # Añadir nodos de orígenes, transbordos y destinos
        sources_list = [f"Source {i+1}" for i in range(len(allocation_stage_one))]
        transhipments_list = [f"Transhipment {i+1}" for i in range(len(allocation_stage_two))]
        destinations_list = [f"Destination {i+1}" for i in range(len(allocation_stage_two[0]))]

        # Añadir los nodos al grafo
        G.add_nodes_from(sources_list, layer='source')
        G.add_nodes_from(transhipments_list, layer='transhipment')
        G.add_nodes_from(destinations_list, layer='destination')

        dict_allocation_stage_one = {}

        # Asignación de orígenes a transbordos
        for i in range(len(allocation_stage_one)):
            list_row = []
            for j in range(len(allocation_stage_one[0])):
                if allocation_stage_one[i][j] != 0:  # Si hay asignación
                    list_row.append(transhipments_list[j])
            dict_allocation_stage_one[sources_list[i]] = list_row

        dict_allocation_stage_two = {}

        # Asignación de transbordos a destinos
        for i in range(len(allocation_stage_two)):
            list_row = []
            for j in range(len(allocation_stage_two[0])):
                if allocation_stage_two[i][j] != 0:  # Si hay asignación
                    list_row.append(destinations_list[j])
                dict_allocation_stage_two[transhipments_list[i]] = list_row

        pprint(dict_allocation_stage_one)
        pprint(dict_allocation_stage_two)
       
        # pprint(allocation_stage_one)
        # pprint(allocation_stage_two)

        # Añadir aristas desde orígenes a transbordos
        for source, transhipments in dict_allocation_stage_one.items():
            for transhipment in transhipments:
                G.add_edge(source, transhipment)

        # Añadir aristas desde transbordos a destinos
        for transhipment, destinations in dict_allocation_stage_two.items():
            for destination in destinations:
                G.add_edge(transhipment, destination)

        # Posiciones de los nodos
        pos = {}

         # Definir coordenadas de los orígenes en x = 0
        for i, source in enumerate(sources_list):
            pos[source] = (0, len(sources_list) - 1 - i)
        
        # Definir coordenadas de los transbordos en x = 1 (centrados entre los orígenes)
        for i, transhipment in enumerate(transhipments_list):
            pos[transhipment] = (1, len(transhipments_list) - 1 - i)
        
        # Definir coordenadas de los destinos en x = 2 (distribuidos uniformemente)
        for i, destination in enumerate(destinations_list):
            pos[destination] = (2, len(destinations_list) - 1 - i)
        pprint(pos)
        # Definir colores para las aristas
        edge_colors = ['red']  # Cambia los colores como prefieras

        # Dibujar nodos
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)

        # Dibujar etiquetas
        nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold')

        # Dibujar aristas con diferentes colores
        nx.draw_networkx_edges(G, pos, edge_color='red', arrows=True, arrowstyle='->', arrowsize=20)

        # Mostrar el grafo
        plt.show()

if __name__ == "__main__":
    # Definir las entradas
    # costs = [[3, 2, 1], [4, 3, 2], [5, 4, 3]]  # Matriz de costos
    # supply = [(10, 12, 14), (15, 17, 19), (20, 22, 24)]  # Números fuzzy triangulares para la oferta
    # demand = [(12, 14, 16), (18, 20, 22), (15, 17, 19)]  # Números fuzzy triangulares para la demanda
    tfn = TriangularFuzzyNumber()
    fuzzy_chart = FuzzyCharts()
    net = Network()
    

    # supply = [  
    #             [(10, 12, 14), (15, 17, 19), (20, 22, 24)], # Ejemplo del libro
    #             [tfn.generate_triangular_fuzzy_number(10, 24), tfn.generate_triangular_fuzzy_number(10, 24), tfn.generate_triangular_fuzzy_number(10, 24)] 
    #             ]  # Números fuzzy triangulares para la oferta

    # demand = [  # Números fuzzy triangulares para la demanda    
    #             [(12, 14, 16), (18, 20, 22), (15, 17, 19)],  # Producto 1
    #             [tfn.generate_triangular_fuzzy_number(12, 22), tfn.generate_triangular_fuzzy_number(12, 22)]  # Producto 2
    #         ] 
    # new_tfn = (0, 0, 0)  
    # sum_supply_by_row = []
    
    # for i in range(len(supply)):
    #     a, b, c = 0, 0, 0
    #     for j in range(len(supply[0])):
    #         a += supply[i][j][0]
    #         b += supply[i][j][1]
    #         c += supply[i][j][2]
    #     sum_supply_by_row.append((a, b, c))

    # sum_demand_by_row = []
    # for i in range(len(demand)):
    #     a, b, c = 0, 0, 0
    #     for j in range(len(demand[1])):
    #         a += demand[i][j][0]
    #         b += demand[i][j][1]
    #         c += demand[i][j][2]
    #     sum_demand_by_row.append((a, b, c))
        
    # for i in range(1, len(demand)):  
    #     new_tfn = sum_supply_by_row[i][0] - sum_demand_by_row[i][0], sum_supply_by_row[i][1] - sum_demand_by_row[i][1], sum_supply_by_row[i][2] - sum_demand_by_row[i][2]
    #     demand[i] = demand[i] + [new_tfn]
    
    # pprint(supply)
    # pprint(demand)
    supply = [
                [(10, 12, 14), (15, 17, 19), (20, 22, 24)],
                [(14, 15, 19), (12, 15, 17), (16, 17, 19)]
            ]
    
    demand = [
                [(12, 14, 16), (18, 20, 22), (15, 17, 19)],
                [(16, 17, 19), (16, 17, 18), (10, 13, 18)]
            ]
    
    
    # products = 2
    # transhipments = 3

    # costs_stage_one = np.zeros((products, len(supply[0]), transhipments))  
    # costs_stage_two = np.zeros((products, transhipments, len(demand[0])))
    
    # for p in range(products):
    #     for i in range(len(supply[0])):
    #         for k in range(transhipments):
    #             costs_stage_one[p][i][k] = random.randint(1, 25)
    # for p in range(products):
    #     for k in range(transhipments):
    #         for j in range(len(demand[0])):
    #             costs_stage_two[p][k][j] = random.randint(1, 25)
                
    # pprint(costs_stage_one)
    # pprint(costs_stage_two)

    costs_stage_one = [
                        [
                            [ 6., 18.,  6.],
                            [ 6., 23., 22.],
                            [ 2., 12., 15.]
                        ],
                        [
                            [24.,  7.,  4.],
                            [23., 14.,  5.],
                            [19.,  9., 14.]
                        ]
                    ]
    costs_stage_two = [
                        [
                            [ 9., 19., 22.],
                            [ 4., 15., 15.],
                            [10., 14.,  2.]
                        ],

                        [
                            [ 1.,  1., 18.],
                            [ 3., 20., 22.],
                            [17.,  4., 24.]
                        ]
                    ]

    solution_list = []
    
    products = len(costs_stage_one)
    sources = len(costs_stage_one[0])
    transhipments = len(costs_stage_two[0])
    destinations = len(costs_stage_two[0][0])

    solutions = 4
    tfn = TriangularFuzzyNumber()

    for z in range(solutions):
        # Crear el problema de transporte fuzzy
        ftp = FuzzyTransportProblem(costs_stage_one, costs_stage_two, supply, demand, z)
        
        matrix_allocation_stage_one = np.zeros_like(costs_stage_one)
        matrix_allocation_stage_two = np.zeros_like(costs_stage_two)

        ftp.create_variables()
        
        ftp.add_constraints()
        ftp.set_objective()
        status = ftp.solve()

        if status == pulp.LpStatusOptimal: 
            solution_stage_one, solution_stage_two = ftp.get_solution()
            
            for (p, i, k), value in solution_stage_one.items():
                matrix_allocation_stage_one[p][i][k] = value

            for (p, k, j), value in solution_stage_two.items():
                matrix_allocation_stage_two[p][k][j] = value

            solution_list.append([matrix_allocation_stage_one, matrix_allocation_stage_two])
            pprint(matrix_allocation_stage_one)
            pprint(matrix_allocation_stage_two)
        else:
            print(pulp.LpSolution[pulp.LpSolutionInfeasible])    
            sys.exit()
    # pprint(solution_list)
    matrix_result_stage_one = np.zeros_like(costs_stage_one, dtype=object)
    matrix_result_stage_two = np.zeros_like(costs_stage_two, dtype=object)
    
    for p in range(products):
        for i in range(sources):
            for k in range(transhipments):
                lower = float((solution_list[1][0][p][i][k] - solution_list[0][0][p][i][k])*0 + solution_list[0][0][p][i][k])
                upper = float((solution_list[3][0][p][i][k] - solution_list[2][0][p][i][k])*0 + solution_list[2][0][p][i][k])
                middle = float(max((solution_list[1][0][p][i][k] - solution_list[0][0][p][i][k])*1 + solution_list[0][0][p][i][k], (solution_list[3][0][p][i][k] - solution_list[2][0][p][i][k])*1 + solution_list[2][0][p][i][k]))
                matrix_result_stage_one[p][i][k] = (lower, middle, upper)

    for p in range(products):
        for k in range(transhipments):
            for j in range(destinations):
                lower = float((solution_list[1][1][p][k][j] - solution_list[0][1][p][k][j])*0 + solution_list[0][1][p][k][j])
                upper = float((solution_list[3][1][p][k][j] - solution_list[2][1][p][k][j])*0 + solution_list[2][1][p][k][j])
                middle = float(max((solution_list[1][1][p][k][j] - solution_list[0][1][p][k][j])*1 + solution_list[0][1][p][k][j], (solution_list[3][1][p][k][j]- solution_list[2][1][p][k][j])*1 + solution_list[2][1][p][k][j]))
                
                matrix_result_stage_two[p][k][j] = (lower, middle, upper)

    
    matrix_result_costs_stage_one = np.zeros_like(costs_stage_one, dtype=object)
    matrix_result_costs_stage_two = np.zeros_like(costs_stage_two, dtype=object)
    # pprint(matrix_result_stage_one)
    # pprint(matrix_result_stage_two)
    for p in range(products):
        for i in range(sources):
            for k in range(transhipments):
                matrix_result_costs_stage_one[p][i][k] = tfn.constant_multiplication(costs_stage_one[p][i][k], matrix_result_stage_one[p][i][k])
    
    for p in range(products):
        for k in range(transhipments):
            for j in range(destinations):
                matrix_result_costs_stage_two[p][k][j] = tfn.constant_multiplication(costs_stage_two[p][k][j], matrix_result_stage_two[p][k][j])


    list_result = [matrix_result_costs_stage_one, matrix_result_costs_stage_two]
    # pprint(list_result)
    optimus_solution_stage_one = (0, 0, 0)
    optimus_solution_stage_two = (0, 0, 0)
    list_solution_stage_one = []
    list_solution_stage_two = []
    list_solution = []
    for p in range(products):
        for i in range(sources):
            for k in range(transhipments):
                optimus_solution_stage_one = tfn.sum_triangular_fuzzy_number(optimus_solution_stage_one, list_result[0][p][i][k])
        list_solution_stage_one.append(optimus_solution_stage_one)

    for p in range(products):
        for k in range(transhipments):
            for j in range(destinations):
    
                    optimus_solution_stage_two = tfn.sum_triangular_fuzzy_number(optimus_solution_stage_two, list_result[1][p][k][j])
        list_solution_stage_two.append(optimus_solution_stage_two)
    
    for i in range(len(list_solution_stage_one)-1, 0, -1):
        list_solution_stage_one[i] = (list_solution_stage_one[i][0] - list_solution_stage_one[i-1][0], list_solution_stage_one[i][1] - list_solution_stage_one[i-1][1], list_solution_stage_one[i][2] - list_solution_stage_one[i-1][2])
    
    for i in range(len(list_solution_stage_two)-1, 0, -1):
        list_solution_stage_two[i] = (list_solution_stage_two[i][0] - list_solution_stage_two[i-1][0], list_solution_stage_two[i][1] - list_solution_stage_two[i-1][1], list_solution_stage_two[i][2] - list_solution_stage_two[i-1][2])

    for i, j in zip(list_solution_stage_one, list_solution_stage_two):
        list_solution.append(tfn.sum_triangular_fuzzy_number(i, j))


    optimus_solution = tfn.sum_triangular_fuzzy_number(optimus_solution_stage_one, optimus_solution_stage_two)
    
    
    # total_supply = []
    # total_demand = []

    # for p in range(products):
    #     sum_supply = (0, 0, 0)
    #     for i in range(sources):
    #         sum_supply = tfn.sum_triangular_fuzzy_number(sum_supply, supply[p][i])
    #     total_supply.append(sum_supply)

    # for p in range(products):
    #     sum_demand = (0, 0, 0)
    #     for j in range(destinations):
    #         sum_demand = tfn.sum_triangular_fuzzy_number(sum_demand, demand[p][j])
    #     total_demand.append(sum_demand)

    print(optimus_solution)
    
    fuzzy_chart.show_chart(supply)
    #net.show_network(matrix_allocation_stage_one[0], matrix_allocation_stage_two[0])

