import pulp
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import networkx as nx
import random

# Clase para manejar números triangulares fuzzy
class LRFlatFuzzyNumber:
    
    # def parametric_form(self, tuple_tfn):
    #     lower = tuple_tfn[0]
    #     middle = tuple_tfn[1]
    #     upper = tuple_tfn[2]
    #     lower_param = [lower + (middle - lower) * 0, lower + (middle - lower) * 1]
    #     upper_param = [upper - (upper - middle) * 0, upper - (upper - middle) * 1]
    #     return lower_param + upper_param
    
    def sum_flat_fuzzy_number(self, tuple1, tuple2):
        tuple3 = (tuple1[0]+tuple2[0], tuple1[1]+tuple2[1], tuple1[2]+tuple2[2], tuple1[3]+tuple2[3])
        return tuple3
    
    def rest_flat_fuzzy_number(tuple1, tuple2):
        tuple3 = (tuple1[0]-tuple2[3], tuple1[1]-tuple2[2], tuple1[2]-tuple2[1], tuple1[3]-tuple2[0])
        return tuple3

    def constant_multiplication(self, k, tuple1):
        tuple2 = (tuple1[0]*k, tuple1[1]*k, tuple1[2]*k, tuple1[3]*k)
        return tuple2
    
    def mult_flat_fuzzy_number(self, tuple1, tuple2):
        tuple3 = (tuple1[0]*tuple2[0], tuple1[1]*tuple2[1], tuple1[2]*tuple2[2], tuple1[3]*tuple2[3])
        return tuple3
    
    def generate_lr_flat_fuzzy_number(self, lower_limit=0, upper_limit=100):

        # Step 1: Generate random numbers using Gaussian distribution for smoother randomness
        base_numbers = [random.gauss((upper_limit + lower_limit) / 2, (upper_limit - lower_limit) / 6) for _ in range(4)]
        
        # Step 2: Sort the numbers to satisfy the trapezoidal condition
        base_numbers.sort()
        
        # Step 3: Apply constraints to adjust differences between them (optional)
        min_distance = round((upper_limit - lower_limit) * 0.1, 0)  # Ensure a minimum distance between points
        
        for i in range(1, 4):
            if base_numbers[i] - base_numbers[i - 1] < min_distance:
                base_numbers[i] = base_numbers[i - 1] + min_distance
        
        # Step 4: Ensure the numbers are within the bounds [a_min, a_max]
        base_numbers = [int(max(min(x, upper_limit), lower_limit)) for x in base_numbers]

        return tuple(base_numbers)
    
# Clase para manejar el problema de optimización usando PuLP
class FuzzyTransportProblemEbrahimnejad:
    def __init__(self, costs_stage_one, costs_stage_two, supply, demand, z):
        self.costs_stage_one = costs_stage_one  
        self.costs_stage_two = costs_stage_two  
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.z = z
        self.sources = len(supply[0])
        self.destinations = len(demand[0])
        self.products = len(self.costs_stage_one)
        self.transhipments = len(self.costs_stage_two[0])
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.ffn = LRFlatFuzzyNumber()
    
    def create_variables(self):
        self.x = pulp.LpVariable.dicts("x", [(p, i, k) for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)], lowBound=0)
        self.y = pulp.LpVariable.dicts("y", [(p, k, j) for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)], lowBound=0)

    def add_constraints(self):
        # Restricciones de oferta
        for p in range(self.products):
            for i in range(self.sources):
                
                pprint(self.supply[p][i][self.z])
                self.problem += pulp.lpSum([self.x[p, i, k] for k in range(self.transhipments)]) == self.supply[p][i][self.z]
            
        # Restricciones de demanda
        for p in range(self.products):
            for j in range(self.destinations):
            
                self.problem += pulp.lpSum([self.y[p, k, j] for k in range(self.transhipments)]) == self.demand[p][j][self.z]

        for p in range(self.products):
            for k in range(self.transhipments):

                self.problem += pulp.lpSum([self.x[p, i, k] for i in range(self.sources)]) == pulp.lpSum([self.y[p, k, j] for j in range(self.destinations)])


    def set_objective(self):
        self.problem += pulp.lpSum(self.costs_stage_one[p][i][k][self.z] * self.x[p, i, k] for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)) + pulp.lpSum(self.costs_stage_two[p][k][j][self.z] * self.y[p, k, j] for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations))
    
    def solve(self):
        self.problem.solve()
        
        return self.problem.status

    def get_solution(self):


        solution_stage_one = {(p, i, k): self.x[p, i, k].varValue for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)}
        solution_stage_two = {(p, k, j): self.y[p, k, j].varValue for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)}
        
        return [solution_stage_one, solution_stage_two]


class FuzzyTransportProblemKaurAndKumar:

    def __init__(self, supply, demand, costs_stage_one, costs_stage_two):
        self.costs_stage_one = costs_stage_one  
        self.costs_stage_two = costs_stage_two  
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.sources = len(supply[0])
        self.destinations = len(demand[0])
        self.products = len(self.costs_stage_one)
        self.transhipments = len(self.costs_stage_two[0])
        self.elements = len(self.supply[0][0])
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.ffn = LRFlatFuzzyNumber()
    
    def create_variables(self):
        self.x = pulp.LpVariable.dicts("x", [(p, i, k, e) for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments) for e in range(self.elements)], lowBound=0)
        self.y = pulp.LpVariable.dicts("y", [(p, k, j, e) for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations) for e in range(self.elements)], lowBound=0)
    
    def add_constraints(self):
        # Restricciones de oferta
        for p in range(self.products):
            for i in range(self.sources):
                for e in range(self.elements):
                
                    self.problem += pulp.lpSum([self.x[p, i, k, e] for k in range(self.transhipments)]) == self.supply[p][i][e]
            
        # Restricciones de demanda
        for p in range(self.products):
            for j in range(self.destinations):
                for e in range(self.elements):
            
                    self.problem += pulp.lpSum([self.y[p, k, j, e] for k in range(self.transhipments)]) == self.demand[p][j][e]

        for p in range(self.products):
            for k in range(self.transhipments):
                for e in range(self.elements):

                    self.problem += pulp.lpSum([self.x[p, i, k, e] for i in range(self.sources)]) == pulp.lpSum([self.y[p, k, j, e] for j in range(self.destinations)])


    def set_objective(self):
        self.problem += pulp.lpSum(self.costs_stage_one[p][i][k][e] * self.x[p, i, k, e] for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments) for e in range(self.elements)) + pulp.lpSum(self.costs_stage_two[p][k][j][e] * self.y[p, k, j, e] for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations) for e in range(self.elements))
    
    def solve(self):
        self.problem.solve()
        
        return self.problem.status

    def get_solution(self):


        solution_stage_one = {(p, i, k, e): self.x[p, i, k, e].varValue for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments) for e in range(self.elements)}
        solution_stage_two = {(p, k, j, e): self.y[p, k, j, e].varValue for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations) for e in range(self.elements)}
        
        return [solution_stage_one, solution_stage_two]




class ApplicationEbrahimnejad:
    

    def __init__(self, supply, demand, costs_stage_one, costs_stage_two):
        
        self.supply = supply
        self.demand = demand
        self.costs_stage_one = costs_stage_one
        self.costs_stage_two = costs_stage_two
        self.ffn = LRFlatFuzzyNumber()

    def run_linear_programs(self):
        solution_list = []
        
        solutions = 4
        

        for z in range(solutions):
            # Crear el problema de transporte fuzzy
            ftpe = FuzzyTransportProblemEbrahimnejad(self.costs_stage_one, self.costs_stage_two, self.supply, self.demand, z)
            
            matrix_result_stage_one = np.zeros((len(self.costs_stage_one), len(self.costs_stage_one[0]), len(self.costs_stage_one[0][0])))
            matrix_result_stage_two = np.zeros((len(self.costs_stage_one), len(self.costs_stage_two[0]), len(self.costs_stage_two[0][0])))

            ftpe.create_variables()
            ftpe.add_constraints()
            ftpe.set_objective()
            status = ftpe.solve()

            if status == pulp.LpStatusOptimal: 
                
                solution_stage_one, solution_stage_two = ftpe.get_solution()
            
                for (p, i, k), value in solution_stage_one.items():
                    matrix_result_stage_one[p][i][k] = value
                    

                for (p, k, j), value in solution_stage_two.items():
                    matrix_result_stage_two[p][k][j] = value

                solution_list.append([matrix_result_stage_one, matrix_result_stage_two])
        pprint(solution_list)
        return solution_list

    def join_linear_programs(self, solution_list):

        
        self.matrix_result_stage_one = np.zeros((len(self.costs_stage_one), len(self.costs_stage_one[0]), len(self.costs_stage_one[0][0])), dtype=object)
        self.matrix_result_stage_two = np.zeros((len(self.costs_stage_one), len(self.costs_stage_two[0]), len(self.costs_stage_two[0][0])), dtype=object)

        
        for p in range(len(self.costs_stage_one)):
            for i in range(len(self.costs_stage_one[0])):
                for k in range(len(self.costs_stage_one[0][0])):
                    a1 = solution_list[0][0][p][i][k]
                    a2 = solution_list[1][0][p][i][k]
                    a3 = solution_list[2][0][p][i][k]
                    a4 = solution_list[3][0][p][i][k]

                    self.matrix_result_stage_one[p][i][k] = (float(a1), float(a2), float(a3), float(a4))
        
        for p in range(len(self.costs_stage_two)):
            for k in range(len(self.costs_stage_two[0])):
                for j in range(len(self.costs_stage_two[0][0])):
                    a1 = solution_list[0][1][p][k][j]
                    a2 = solution_list[1][1][p][k][j]
                    a3 = solution_list[2][1][p][k][j]
                    a4 = solution_list[3][1][p][k][j]

                    self.matrix_result_stage_two[p][k][j] = (float(a1), float(a2), float(a3), float(a4))
        
        matrix_result_costs_stage_one = np.zeros((len(self.costs_stage_one), len(self.costs_stage_one[0]), len(self.costs_stage_one[0][0])), dtype=object)
        matrix_result_costs_stage_two = np.zeros((len(self.costs_stage_one), len(self.costs_stage_two[0]), len(self.costs_stage_two[0][0])), dtype=object)

        for p in range(len(self.costs_stage_one)):
            for i in range(len(self.costs_stage_one[0])):
                for k in range(len(self.costs_stage_one[0][0])):
                    matrix_result_costs_stage_one[p][i][k] = self.ffn.mult_flat_fuzzy_number(self.costs_stage_one[p][i][k], self.matrix_result_stage_one[p][i][k])

        for p in range(len(self.costs_stage_two)):
            for k in range(len(self.costs_stage_two[0])):
                for j in range(len(self.costs_stage_two[0][0])):
                    matrix_result_costs_stage_two[p][k][j] = self.ffn.mult_flat_fuzzy_number(self.costs_stage_two[p][k][j], self.matrix_result_stage_two[p][k][j])

        # pprint(matrix_result_costs)
        return matrix_result_costs_stage_one, matrix_result_costs_stage_two

    def get_optimus(self, matrix_results_costs_stage_one, matrix_results_costs_stage_two):
        self.optimus_solution_stage_one = (0, 0, 0, 0)
        self.optimus_solution_stage_two = (0, 0, 0, 0)

        for p in range(len(self.costs_stage_one)):
            for i in range(len(self.costs_stage_one[0])):
                for k in range(len(self.costs_stage_one[0][0])):
                    self.optimus_solution_stage_one = self.ffn.sum_flat_fuzzy_number(self.optimus_solution_stage_one, matrix_results_costs_stage_one[p][i][k])
        
        for p in range(len(self.costs_stage_two)):
            for k in range(len(self.costs_stage_two[0])):
                for j in range(len(self.costs_stage_two[0][0])):
                    self.optimus_solution_stage_two = self.ffn.sum_flat_fuzzy_number(self.optimus_solution_stage_two, matrix_results_costs_stage_two[p][k][j])

        optimus_solution = self.ffn.sum_flat_fuzzy_number(self.optimus_solution_stage_one, self.optimus_solution_stage_two)
        # print(optimus_solution)
        return optimus_solution


class ApplicationKaurAndKumar:
    

    def __init__(self, supply, demand, costs_stage_one, costs_stage_two):
        
        self.supply = supply
        self.demand = demand
        self.costs_stage_one = costs_stage_one
        self.costs_stage_two = costs_stage_two
        self.ffn = LRFlatFuzzyNumber()

    def run_linear_programs(self):
        
        

    
        # Crear el problema de transporte fuzzy
        ftpk = FuzzyTransportProblemKaurAndKumar(self.supply, self.demand, self.costs_stage_one, self.costs_stage_two)
        
        matrix_result_stage_one = np.zeros((len(self.costs_stage_one), len(self.costs_stage_one[0]), len(self.costs_stage_one[0][0])), dtype=object)
        matrix_result_stage_two = np.zeros((len(self.costs_stage_one), len(self.costs_stage_two[0]), len(self.costs_stage_two[0][0])), dtype=object)

        ftpk.create_variables()
        ftpk.add_constraints()
        ftpk.set_objective()
        status = ftpk.solve()

        if status == pulp.LpStatusOptimal: 
            solution_stage_one, solution_stage_two = ftpk.get_solution()
            tuple_result = [0, 0, 0, 0]
            for (p, i, k, e), value in solution_stage_one.items():
                
                tuple_result[e] = value
                
                if e == 3:
                    tuple_result = tuple(tuple_result)

                    matrix_result_stage_one[p][i][k] = tuple_result

                    tuple_result = [0, 0, 0, 0]

            for (p, k, j, e), value in solution_stage_two.items():
                
                tuple_result[e] = value
                
                if e == 3:
                    tuple_result = tuple(tuple_result)

                    matrix_result_stage_two[p][k][j] = tuple_result

                    tuple_result = [0, 0, 0, 0]
                    
        return matrix_result_stage_one, matrix_result_stage_two
            
         

    def join_linear_programs(self, matrix_result_stage_one, matrix_result_stage_two):

        
        matrix_result_costs_stage_one = np.zeros((len(self.costs_stage_one), len(self.costs_stage_one[0]), len(self.costs_stage_one[0][0])), dtype=object)
        matrix_result_costs_stage_two = np.zeros((len(self.costs_stage_one), len(self.costs_stage_two[0]), len(self.costs_stage_two[0][0])), dtype=object)

        
        for p in range(len(self.costs_stage_one)):
            for i in range(len(self.costs_stage_one[0])):
                for k in range(len(self.costs_stage_one[0][0])):
                    matrix_result_costs_stage_one[p][i][k] = self.ffn.mult_flat_fuzzy_number(self.costs_stage_one[p][i][k], matrix_result_stage_one[p][i][k])

        for p in range(len(self.costs_stage_two)):
            for k in range(len(self.costs_stage_two[0])):
                for j in range(len(self.costs_stage_two[0][0])):
                    matrix_result_costs_stage_two[p][k][j] = self.ffn.mult_flat_fuzzy_number(self.costs_stage_two[p][k][j], matrix_result_stage_two[p][k][j])

        # pprint(matrix_result_costs)
        return matrix_result_costs_stage_one, matrix_result_costs_stage_two

    def get_optimus(self, matrix_result_costs_stage_one, matrix_result_costs_stage_two):
        optimus_solution_stage_one = (0, 0, 0, 0)
        optimus_solution_stage_two = (0, 0, 0, 0)

        for p in range(len(self.costs_stage_one)):
            for i in range(len(self.costs_stage_one[0])):
                for k in range(len(self.costs_stage_one[0][0])):
                    
                    optimus_solution_stage_one = self.ffn.sum_flat_fuzzy_number(optimus_solution_stage_one, matrix_result_costs_stage_one[p][i][k])
        
        for p in range(len(self.costs_stage_two)):
            for k in range(len(self.costs_stage_two[0])):
                for j in range(len(self.costs_stage_two[0][0])):
                    
                    optimus_solution_stage_two = self.ffn.sum_flat_fuzzy_number(optimus_solution_stage_two, matrix_result_costs_stage_two[p][k][j])

        optimus_solution = self.ffn.sum_flat_fuzzy_number(optimus_solution_stage_one, optimus_solution_stage_two)
        # print(optimus_solution)
        return optimus_solution

class FuzzyCharts():
    
    
    def show_chart(self, list_tfn):
        i = 0
        self.list_tfn = list_tfn
        pprint(self.list_tfn)
        # Configurar matplotlib para usar LaTeX
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')  # Usar Computer Modern, que es la fuente por defecto de LaTeX

        x = np.linspace(min([solution[0] for solution in self.list_tfn if solution[0] != 0]) - 5, max([solution[3] for solution in self.list_tfn]) + 5, 500)

        list_a = [solution[0] for solution in self.list_tfn]
        list_b = [solution[1] for solution in self.list_tfn]
        list_c = [solution[2] for solution in self.list_tfn]
        list_d = [solution[3] for solution in self.list_tfn]
        # Definir la función de pertenencia trapezoidal
        

        # Calcular los grados de pertenencia
        y = np.maximum(np.minimum(np.minimum((x-list_a[0])/(list_b[0]-list_a[0]), 1), (list_d[0]-x)/(list_d[0]-list_c[0])), 0)
        z = np.maximum(np.minimum(np.minimum((x-list_a[1])/(list_b[1]-list_a[1]), 1), (list_d[1]-x)/(list_d[1]-list_c[1])), 0)
        w = np.maximum(np.minimum(np.minimum((x-list_a[2])/(list_b[2]-list_a[2]), 1), (list_d[2]-x)/(list_d[2]-list_c[2])), 0)
        # u = np.maximum(np.minimum(np.minimum((x-list_a[3])/(list_b[3]-list_a[3]), 1), (list_d[3]-x)/(list_d[3]-list_c[3])), 0)


        # Graficar
        plt.plot(x, y, label=f'Óptimo orígenes a trasbordos ({list_a[0]}, {list_b[0]}, {list_c[0]}, {list_d[0]})')
        plt.plot(x, z, label=f'Óptimo trasbordos a destinos ({list_a[1]}, {list_b[1]}, {list_c[1]}, {list_d[1]})')
        plt.plot(x, w, label=f'Óptimo global ({list_a[2]}, {list_b[2]}, {list_c[2]}, {list_d[2]})')
        # plt.plot(x, u, label=f'Producto 2 - Trasbordo 2 ({list_a[3]}, {list_b[3]}, {list_c[3]}, {list_d[3]})')

        plt.fill_between(x, y, alpha=0.2)
        plt.fill_between(x, z, alpha=0.2)
        plt.fill_between(x, w, alpha=0.2)
        # plt.fill_between(x, u, alpha=0.2)
        plt.xlabel(r'Unidades monetarias (\$)')
        plt.ylabel(r'Grado de pertenencia $\alpha$')
        plt.title(f'Valores óptimos')
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
                if allocation_stage_one[i][j] != (0, 0, 0, 0):  # Si hay asignación
                    list_row.append(transhipments_list[j])
            dict_allocation_stage_one[sources_list[i]] = list_row

        dict_allocation_stage_two = {}

        # Asignación de transbordos a destinos
        for i in range(len(allocation_stage_two)):
            list_row = []
            for j in range(len(allocation_stage_two[0])):
                if allocation_stage_two[i][j] != (0, 0, 0, 0):  # Si hay asignación
                    list_row.append(destinations_list[j])
                dict_allocation_stage_two[transhipments_list[i]] = list_row

        # pprint(dict_allocation_stage_one)
        # pprint(dict_allocation_stage_two)
       
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
        # pprint(pos)
        # Definir colores para las aristas
        edge_colors = ['red']  # Cambia los colores como prefieras

        # Dibujar nodos
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)

        # Dibujar etiquetas
        nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold')

        # Dibujar aristas con diferentes colores
        nx.draw_networkx_edges(G, pos, edge_color='red', arrows=True, arrowstyle='->', arrowsize=20)

        # Mostrar el grafo
        plt.title('Red óptima producto 2')
        plt.show()

if __name__ == "__main__":

    lrfn = LRFlatFuzzyNumber()
    # supply = [
    #             [lrfn.generate_lr_flat_fuzzy_number(2475, 4000), lrfn.generate_lr_flat_fuzzy_number(2475, 4000), lrfn.generate_lr_flat_fuzzy_number(2475, 4000)],
    #             [lrfn.generate_lr_flat_fuzzy_number(2475, 4000), lrfn.generate_lr_flat_fuzzy_number(2475, 4000), lrfn.generate_lr_flat_fuzzy_number(2475, 4000)]
    #         ]   
    # demand = [
    #             [lrfn.generate_lr_flat_fuzzy_number(1950, 3200), lrfn.generate_lr_flat_fuzzy_number(1950, 3200), lrfn.generate_lr_flat_fuzzy_number(1950, 3200)],
    #             [lrfn.generate_lr_flat_fuzzy_number(1950, 3200), lrfn.generate_lr_flat_fuzzy_number(1950, 3200), lrfn.generate_lr_flat_fuzzy_number(1950, 3200)]
    #         ]
    
    # new_tfn = (0, 0, 0, 0)  
    # sum_supply_by_row = []
    
    # for i in range(len(supply)):
    #     a, b, c, d = 0, 0, 0, 0
    #     for j in range(len(supply[0])):
    #         a += supply[i][j][0]
    #         b += supply[i][j][1]
    #         c += supply[i][j][2]
    #         d += supply[i][j][3]
    #     sum_supply_by_row.append((a, b, c, d))

    # sum_demand_by_row = []
    # for i in range(len(demand)):
    #     a, b, c, d = 0, 0, 0, 0
    #     for j in range(len(demand[1])):
    #         a += demand[i][j][0]
    #         b += demand[i][j][1]
    #         c += demand[i][j][2]
    #         d += demand[i][j][3]
    #     sum_demand_by_row.append((a, b, c, d))
        
    # for i in range(len(demand)):  
    #     new_tfn = sum_supply_by_row[i][0] - sum_demand_by_row[i][0], sum_supply_by_row[i][1] - sum_demand_by_row[i][1], sum_supply_by_row[i][2] - sum_demand_by_row[i][2], sum_supply_by_row[i][3
    #                                                                                                                                                                                             ] - sum_demand_by_row[i][3]
    #     demand[i] = demand[i] + [new_tfn]

    # costs_stage_one = [
    #                     # Producto 1
    #                     [
    #                         [lrfn.generate_lr_flat_fuzzy_number(19, 22), lrfn.generate_lr_flat_fuzzy_number(59, 65)], 
    #                         [lrfn.generate_lr_flat_fuzzy_number(97, 105), lrfn.generate_lr_flat_fuzzy_number(15, 21)], 
    #                         [lrfn.generate_lr_flat_fuzzy_number(260, 270), lrfn.generate_lr_flat_fuzzy_number(240, 255)]
    #                     ],
    #                     # Producto 2
    #                     [
    #                         [lrfn.generate_lr_flat_fuzzy_number(230, 250), lrfn.generate_lr_flat_fuzzy_number(49, 59)], 
    #                         [lrfn.generate_lr_flat_fuzzy_number(13, 22), lrfn.generate_lr_flat_fuzzy_number(88, 101)], 
    #                         [lrfn.generate_lr_flat_fuzzy_number(250, 265), lrfn.generate_lr_flat_fuzzy_number(17, 23)]
    #                     ]
    #                 ]
    
    # costs_stage_two = [
    #                     # Producto 1
    #                     [
    #                         [lrfn.generate_lr_flat_fuzzy_number(97, 105), lrfn.generate_lr_flat_fuzzy_number(15, 21), lrfn.generate_lr_flat_fuzzy_number(110, 119), lrfn.generate_lr_flat_fuzzy_number(190, 240)], 
    #                         [lrfn.generate_lr_flat_fuzzy_number(260, 270), lrfn.generate_lr_flat_fuzzy_number(240, 255), lrfn.generate_lr_flat_fuzzy_number(72, 90), lrfn.generate_lr_flat_fuzzy_number(320, 340)]
    #                     ],
    #                     # Producto 2
    #                     [
    #                         [lrfn.generate_lr_flat_fuzzy_number(320, 340), lrfn.generate_lr_flat_fuzzy_number(260, 280), lrfn.generate_lr_flat_fuzzy_number(215, 238), lrfn.generate_lr_flat_fuzzy_number(72, 95)], 
    #                         [lrfn.generate_lr_flat_fuzzy_number(110, 119), lrfn.generate_lr_flat_fuzzy_number(97, 105), lrfn.generate_lr_flat_fuzzy_number(14, 25), lrfn.generate_lr_flat_fuzzy_number(145, 180)]
    #                     ]
    #                 ]
    # pprint(supply)
    # pprint(demand)
    # pprint(costs_stage_one)
    # pprint(costs_stage_two)

    supply = [
                [(2999, 3151, 3303, 3465), (3179, 3331, 3483, 3805), (2915, 3067, 3419, 3471)],
                [(3394, 3546, 3698, 3850), (2906, 3191, 3407, 3559), (3160, 3412, 3495, 3647)]
            ]
    demand = [
                [(2198, 2323, 2595, 2772), (2460, 2585, 2832, 2957), (2302, 2427, 2552, 2677), (2133, 2214, 2226, 2335)],
                [(2277, 2489, 2642, 2767), (2328, 2631, 2756, 2881), (2471, 2596, 2721, 2865), (2384, 2433, 2481, 2543)]
            ]
    costs_stage_one = [
                        [
                            [(19, 19, 19, 20), (61, 62, 63, 64)],
                            [(99, 101, 102, 103), (16, 17, 18, 19)],
                            [(262, 264, 265, 266), (245, 247, 249, 251)]
                        ],
                        [
                            [(235, 237, 240, 243), (54, 55, 56, 57)],
                            [(16, 18, 19, 20), (89, 94, 95, 97)],
                            [(254, 256, 259, 261), (19, 20, 21, 22)]
                        ]
                    ]
    costs_stage_two = [
                        [
                            [(99, 100, 101, 102), (15, 18, 19, 20), (113, 114, 115, 117), (204, 209, 219, 224)],
                            [(263, 264, 265, 266), (242, 245, 247, 249), (76, 80, 82, 84), (326, 328, 331, 333)]
                        ],
                        [
                            [(320, 329, 331, 333), (267, 269, 271, 274), (223, 225, 227, 229), (79, 83, 86, 89)],
                            [(113, 114, 115, 116), (98, 101, 102, 104), (20, 21, 22, 23), (153, 160, 164, 168)]
                        ]
                    ]

   
    ebrahimnejad = ApplicationEbrahimnejad(supply, demand, costs_stage_one, costs_stage_two)

    solution_list = ebrahimnejad.run_linear_programs()
    # pprint(solution_list)
    matrix_results_costs_stage_one, matrix_result_costs_stage_two = ebrahimnejad.join_linear_programs(solution_list)
    optimus_solution = ebrahimnejad.get_optimus(matrix_results_costs_stage_one, matrix_result_costs_stage_two)

    # kaurandkumar = ApplicationKaurAndKumar(supply, demand, costs_stage_one, costs_stage_two)

    # matrix_result_stage_one, matrix_result_stage_two = kaurandkumar.run_linear_programs()
    # matrix_results_costs_stage_one, matrix_result_costs_stage_two = kaurandkumar.join_linear_programs(matrix_result_stage_one, matrix_result_stage_two)
    # optimus_solution = kaurandkumar.get_optimus(matrix_results_costs_stage_one, matrix_result_costs_stage_two)

    
    pprint(ebrahimnejad.matrix_result_stage_one)

    fuzzy_chart = FuzzyCharts()
    # fuzzy_chart.show_chart([ebrahimnejad.optimus_solution_stage_one, ebrahimnejad.optimus_solution_stage_two, optimus_solution])

    net = Network()
    net.show_network(ebrahimnejad.matrix_result_stage_one[1], ebrahimnejad.matrix_result_stage_two[1])