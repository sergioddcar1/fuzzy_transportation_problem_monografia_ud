import pulp
import numpy as np
from pprint import pprint

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

# Clase para manejar el problema de optimización usando PuLP
class FuzzyTransportProblemEbrahimnejad:
    def __init__(self, costs, supply, demand, z):
        self.costs = costs  # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.z = z
        self.rows = len(supply)
        self.cols = len(demand)
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.ffn = LRFlatFuzzyNumber()
    
    def create_variables(self):
        self.x = pulp.LpVariable.dicts("x", [(i, j) for i in range(self.rows) for j in range(self.cols)], lowBound=0)
    
    def add_constraints(self):
        # Restricciones de oferta
        for i in range(self.rows):
            
            self.problem += pulp.lpSum([self.x[i, j] for j in range(self.cols)]) == self.supply[i][self.z]
        
        # Restricciones de demanda
        for j in range(self.cols):
           
            self.problem += pulp.lpSum([self.x[i, j] for i in range(self.rows)]) == self.demand[j][self.z]
    
    def set_objective(self):
        self.problem += pulp.lpSum(self.costs[i][j][self.z] * self.x[i, j] for i in range(self.rows) for j in range(self.cols))
    
    def solve(self):
        self.problem.solve()
        
        return self.problem.status

    def get_solution(self):
        solution = {(i, j): self.x[i, j].varValue for i in range(self.rows) for j in range(self.cols)}
        return solution



class FuzzyTransportProblemKaurAndKumar:

    def __init__(self, supply, demand, costs):
        self.costs = costs  # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.rows = len(supply)
        self.cols = len(demand)
        self.elements = len(supply[0])
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.ffn = LRFlatFuzzyNumber()
    
    def create_variables(self):
        self.x = pulp.LpVariable.dicts("x", [(i, j, k) for i in range(self.rows) for j in range(self.cols) for k in range(self.elements)], lowBound=0)
    
    def add_constraints(self):
        # Restricciones de oferta
        for i in range(self.rows):
            for k in range(self.elements):
            
                self.problem += pulp.lpSum([self.x[i, j, k] for j in range(self.cols)]) == self.supply[i][k]
        
        # Restricciones de demanda
        for j in range(self.cols):
            for k in range(self.elements):
           
                self.problem += pulp.lpSum([self.x[i, j, k] for i in range(self.rows)]) == self.demand[j][k]
    
    def set_objective(self):
        self.problem += pulp.lpSum(self.costs[i][j][k] * self.x[i, j, k] for i in range(self.rows) for j in range(self.cols) for k in range(self.elements))
    
    def solve(self):
        self.problem.solve()
        
        return self.problem.status

    def get_solution(self):
        solution = {(i, j, k): self.x[i, j, k].varValue for i in range(self.rows) for j in range(self.cols) for k in range(self.elements)}
        return solution




class ApplicationEbrahimnejad:
    

    def __init__(self, supply, demand, costs):
        
        self.supply = supply
        self.demand = demand
        self.costs = costs
        self.ffn = LRFlatFuzzyNumber()

    def run_linear_programs(self):
        solution_list = []
        
        solutions = 4
        

        for z in range(solutions):
            # Crear el problema de transporte fuzzy
            ftpe = FuzzyTransportProblemEbrahimnejad(self.costs, self.supply, self.demand, z)
            
            matrix_result = np.zeros((len(self.supply), len(self.demand)))

            ftpe.create_variables()
            ftpe.add_constraints()
            ftpe.set_objective()
            status = ftpe.solve()

            if status == pulp.LpStatusOptimal: 
                solution = ftpe.get_solution()
                
                for (i, j), value in solution.items():
                    matrix_result[i][j] = value

                solution_list.append(matrix_result)
            
        # pprint(solution_list)
        return solution_list

    def join_linear_programs(self, solution_list):

        
        matrix_result = np.zeros((len(self.supply), len(self.demand)), dtype=object)
        
        for i in range(len(self.supply)):
            for j in range(len(self.demand)):
                a1 = solution_list[0][i][j]
                a2 = solution_list[1][i][j]
                a3 = solution_list[2][i][j]
                a4 = solution_list[3][i][j]

                matrix_result[i][j] = (float(a1), float(a2), float(a3), float(a4))
        # pprint(matrix_result)
        
        matrix_result_costs = np.zeros((len(self.supply), len(self.demand)), dtype=object)
        
        for i in range(len(self.supply)):
            for j in range(len(self.demand)):
                matrix_result_costs[i][j] = self.ffn.mult_flat_fuzzy_number(self.costs[i][j], matrix_result[i][j])

        # pprint(matrix_result_costs)
        return matrix_result_costs

    def get_optimus(self, matrix_result_costs):
        optimus_solution = (0, 0, 0, 0)

        for i in range(len(self.supply)):
            for j in range(len(self.demand)):
                optimus_solution = self.ffn.sum_flat_fuzzy_number(optimus_solution, matrix_result_costs[i][j])
        
        print(optimus_solution)
        return optimus_solution


class ApplicationKaurAndKumar:
    

    def __init__(self, supply, demand, costs):
        
        self.supply = supply
        self.demand = demand
        self.costs = costs
        self.ffn = LRFlatFuzzyNumber()

    def run_linear_programs(self):
        
        

    
        # Crear el problema de transporte fuzzy
        ftpk = FuzzyTransportProblemKaurAndKumar(self.supply, self.demand, self.costs)
        
        matrix_result = np.zeros((len(self.supply), len(self.demand)), dtype=object)

        ftpk.create_variables()
        ftpk.add_constraints()
        ftpk.set_objective()
        status = ftpk.solve()

        if status == pulp.LpStatusOptimal: 
            solution = ftpk.get_solution()
            tuple_result = [0, 0, 0, 0]
            for (i, j, k), value in solution.items():
                
                tuple_result[k] = value
                
                if k == 3:
                    tuple_result = tuple(tuple_result)

                    matrix_result[i][j] = tuple_result

                    tuple_result = [0, 0, 0, 0]
                    
        return matrix_result
            
         

    def join_linear_programs(self, matrix_result):

        
        matrix_result_costs = np.zeros((len(self.supply), len(self.demand)), dtype=object)
        
        for i in range(len(self.supply)):
            for j in range(len(self.demand)):
                matrix_result_costs[i][j] = self.ffn.mult_flat_fuzzy_number(self.costs[i][j], matrix_result[i][j])

        # pprint(matrix_result_costs)
        return matrix_result_costs

    def get_optimus(self, matrix_result_costs):
        optimus_solution = (0, 0, 0, 0)

        for i in range(len(self.supply)):
            for j in range(len(self.demand)):
                optimus_solution = self.ffn.sum_flat_fuzzy_number(optimus_solution, matrix_result_costs[i][j])
        
        print(optimus_solution)
        return optimus_solution



if __name__ == "__main__":


    supply = [(3500, 3555, 3580, 4000), (3125, 3175, 3190, 3200), (2475, 2995, 3275, 3400)]
    demand = [(2050, 2500, 2700, 3050), (3000, 3050, 3100, 3200), (2100, 2150, 2190, 2250), (1950, 2025, 2055, 2100)]
    
    costs = [
                [(19, 20, 21, 22), (59, 62, 63, 65), (90, 95, 97, 99), (150, 160, 165, 170)],
                [(97, 99, 103, 105), (15, 17, 19, 21), (110, 112, 115, 119), (190, 210, 220, 240)],
                [(260, 262, 264, 270), (240, 247, 249, 255), (272, 274, 279, 290), (320, 326, 332, 340)]
            ]
    

    # ebrahimnejad = ApplicationEbrahimnejad(supply, demand, costs)

    # solution_list = ebrahimnejad.run_linear_programs()
    # matrix_results_costs = ebrahimnejad.join_linear_programs(solution_list)
    # optimus_solution = ebrahimnejad.get_optimus(matrix_results_costs)

    kaurandkumar = ApplicationKaurAndKumar(supply, demand, costs)

    matrix_result = kaurandkumar.run_linear_programs()
    matrix_results_costs = kaurandkumar.join_linear_programs(matrix_result)
    optimus_solution = kaurandkumar.get_optimus(matrix_results_costs)