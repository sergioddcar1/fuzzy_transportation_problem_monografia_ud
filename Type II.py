import pulp
import numpy as np
from pprint import pprint

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

    def constant_multiplication(self, k, tuple1):
        tuple2 = (tuple1[0]*k, tuple1[1]*k, tuple1[2]*k)
        return tuple2
    

# Clase para manejar el problema de optimización usando PuLP
class FuzzyTransportProblem:
    def __init__(self, costs, supply, demand, z):
        self.costs = costs  # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.z = z
        self.rows = len(supply)
        self.cols = len(demand)
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.tfn = TriangularFuzzyNumber()
    
    def create_variables(self):
        self.x = pulp.LpVariable.dicts("x", [(i, j) for i in range(self.rows) for j in range(self.cols)], lowBound=0)
    
    def add_constraints(self, fuzzy_supply, fuzzy_demand):
        # Restricciones de oferta
        for i in range(self.rows):
            supply_val = self.tfn.parametric_form(fuzzy_supply[i])
            pprint(supply_val)
            self.problem += pulp.lpSum([self.x[i, j] for j in range(self.cols)]) == supply_val[self.z]
        
        # Restricciones de demanda
        for j in range(self.cols):
            demand_val = self.tfn.parametric_form(fuzzy_demand[j])
            pprint(demand_val)
            self.problem += pulp.lpSum([self.x[i, j] for i in range(self.rows)]) == demand_val[self.z]
    
    def set_objective(self):
        self.problem += pulp.lpSum(self.costs[i][j] * self.x[i, j] for i in range(self.rows) for j in range(self.cols))
    
    def solve(self):
        self.problem.solve()
        
        return self.problem.status

    def get_solution(self):
        solution = {(i, j): self.x[i, j].varValue for i in range(self.rows) for j in range(self.cols)}
        return solution

# # Clase para almacenar y calcular soluciones fuzzy triangulares
# class FuzzySolution:
#     def __init__(self):
#         self.z = 4

#     def loop_solution(self):
        

#     def get_fuzzy_solution(self):
#         fuzzy_solution = {}
#         for key in self.solutions_L[0]:  # Iteramos sobre las variables
#             L_0 = self.solutions_L[0][key]
#             L_1 = self.solutions_L[1][key]
#             U_0 = self.solutions_U[0][key]
#             U_1 = self.solutions_U[1][key]

#             # Generar el número fuzzy triangular
#             fuzzy_solution[key] = (
#                 L_1 - L_0,  # (L(1) - L(0)) * r + L(0)
#                 L_0,        # L(0)
#                 U_1 - U_0,  # (U(1) - U(0)) * r + U(0)
#                 U_0         # U(0)
#             )
#         return fuzzy_solution


    

if __name__ == "__main__":
    # Definir las entradas
    # costs = [[3, 2, 1], [4, 3, 2], [5, 4, 3]]  # Matriz de costos
    # supply = [(10, 12, 14), (15, 17, 19), (20, 22, 24)]  # Números fuzzy triangulares para la oferta
    # demand = [(12, 14, 16), (18, 20, 22), (15, 17, 19)]  # Números fuzzy triangulares para la demanda

    costs = [[8, 6, 12], [5, 7, 10]]  # Matriz de costos
    supply = [(13, 15, 17), (8, 11, 11)]  # Números fuzzy triangulares para la oferta
    demand = [(3, 5, 7), (7, 9, 9), (11, 12, 12)]  # Números fuzzy triangulares para la demanda    

    solution_list = []
    
    solutions = 4
    tfn = TriangularFuzzyNumber()

    for z in range(solutions):
        # Crear el problema de transporte fuzzy
        ftp = FuzzyTransportProblem(costs, supply, demand, z)
        
        matrix_result = np.zeros((len(supply), len(demand)))

        ftp.create_variables()
        ftp.add_constraints(supply, demand)
        ftp.set_objective()
        status = ftp.solve()

        if status == pulp.LpStatusOptimal: 
            solution = ftp.get_solution()
            
            for (i, j), value in solution.items():
                matrix_result[i][j] = value

            solution_list.append(matrix_result)
            
    pprint(solution_list)
    matrix_result = np.zeros((len(supply), len(demand)), dtype=object)
    
    for i in range(len(supply)):
        for j in range(len(demand)):
            lower = float((solution_list[1][i][j] - solution_list[0][i][j])*0 + solution_list[0][i][j])
            upper = float((solution_list[3][i][j] - solution_list[2][i][j])*0 + solution_list[2][i][j])
            middle = float(max((solution_list[1][i][j] - solution_list[0][i][j])*1 + solution_list[0][i][j], (solution_list[3][i][j] - solution_list[2][i][j])*1 + solution_list[2][i][j]))
            
            matrix_result[i][j] = tfn.constant_multiplication(costs[i][j], (lower, middle, upper))
    pprint(matrix_result)
    optimus_solution = (0, 0, 0)

    for i in range(len(supply)):
        for j in range(len(demand)):
            optimus_solution = tfn.sum_triangular_fuzzy_number(optimus_solution, matrix_result[i][j])

    
    
    print(optimus_solution)
