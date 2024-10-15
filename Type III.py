import pulp
import numpy as np
from pprint import pprint
import sys
import random
import matplotlib.pyplot as plt
import networkx as nx
from amplpy import AMPL



class TrapezoidalFuzzyNumber():

    def parametric_form(self, tuple_tfn, alpha):
        a = tuple_tfn[0]
        b = tuple_tfn[1]
        c = tuple_tfn[2]
        d = tuple_tfn[3]
        fuzzy_param = [a + (b - a) * alpha, d - (d - c) * alpha]
        return fuzzy_param

    def sum_trapezoidal_fuzzy_numbers(self, tuple1, tuple2):
        """

        :param tuple1: Tupla de número trapezoidal difuso y valor de pertenencia (el número trapezoidal debe estar compuesto por 4 enteros).
        :param tuple2: Tupla de número trapezoidal difuso y valor de pertenencia (el número trapezoidal debe estar compuesto por 4 enteros).
        :return: Tupla de número trapezoidal difuso sumado y valor de pertenencia mínimo (el número trapezoidal debe estar compuesto por 4 enteros).
        """
        trapezoidal_number1 = tuple1        
        trapezoidal_number2 = tuple2
        # Sumar los primeros 4 elementos del vector 1 con los primeros 4 elementos del vector 2 en orden invertido
        summed_trapezoidal_number = [trapezoidal_number1[i] + trapezoidal_number2[i] for i in range(len(trapezoidal_number1))]

       

        # Crear el vector de resultado de la suma
        final_tuple = tuple(summed_trapezoidal_number)

        return final_tuple
    
    def rest_trapezoidal_fuzzy_numbers(self, tuple1, tuple2):
        """

        :param tuple1: Tupla de número trapezoidal difuso y valor de pertenencia (el número trapezoidal debe estar compuesto por 4 enteros y la pertenencia debe estar entre 0 y 1).
        :param tuple2: Tupla de número trapezoidal difuso y valor de pertenencia (el número trapezoidal debe estar compuesto por 4 enteros y la pertenencia debe estar entre 0 y 1).
        :return: Tupla de número trapezoidal difuso restado y valor de pertenencia mínimo (el número trapezoidal debe estar compuesto por 4 enteros y la pertenencia debe estar entre 0 y 1).
        """

        trapezoidal_number1 = tuple1        
        trapezoidal_number2 = tuple2
        # Sumar los primeros 4 elementos del vector 1 con los primeros 4 elementos del vector 2 en orden invertido
        rested_trapezoidal_number = [trapezoidal_number1[i] - trapezoidal_number2[len(trapezoidal_number1)-1-i] for i in range(len(trapezoidal_number1))]


        # Crear el vector de resultado de la suma
        final_tuple = tuple(rested_trapezoidal_number)

        return final_tuple


    def constant_multiplication(self, constant, tuple):
        trapezoidal_fuzzy_number = tuple

        trapezoidal_fuzzy_number = [constant*i for i in trapezoidal_fuzzy_number]

        return tuple(trapezoidal_fuzzy_number)

    def generate_trapezoidal_fuzzy_number(self, lower_limit=0, upper_limit=100):

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
    
class FuzzyPrimalTransportProblem:
    def __init__(self, costs, supply, demand):
        self.costs = costs  # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.rows = len(supply)
        self.cols = len(demand)
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.tfn = TrapezoidalFuzzyNumber()
    def parametrize_fuzzy_numbers(self, alpha):
        self.param_costs = np.zeros((self.rows, self.cols), dtype=object)
        self.param_supply = np.zeros((self.rows), dtype=object)
        self.param_demand = np.zeros((self.cols), dtype=object)

        for i in range(self.rows):
            for j in range(self.cols):
                self.param_costs[i][j] = self.tfn.parametric_form(self.costs[i][j], alpha)
                if i == self.rows - 1:
                    self.param_demand[j] = self.tfn.parametric_form(self.demand[j], alpha)
                    
            self.param_supply[i] = self.tfn.parametric_form(self.supply[i], alpha)

        pprint(self.param_supply)
        pprint(self.param_demand)
    def create_variables(self):
        self.x = pulp.LpVariable.dicts("x", [(i, j) for i in range(self.rows) for j in range(self.cols)], lowBound=0)
        self.a = pulp.LpVariable.dicts("a", [(i) for i in range(self.rows)])
        self.b = pulp.LpVariable.dicts("b", [(j) for j in range(self.cols)])
    
    def add_constraints(self):
        # Restricciones de oferta
        for i in range(self.rows):
            
            self.problem += pulp.lpSum([self.x[i, j] for j in range(self.cols)]) == self.a[i]
        
        # Restricciones de demanda
        for j in range(self.cols):
            
            self.problem += pulp.lpSum([self.x[i, j] for i in range(self.rows)]) == self.b[j]

        self.problem += pulp.lpSum([self.a[i] for i in range(self.rows)]) == pulp.lpSum([self.b[j] for j in range(self.cols)])
    
        for i in range(self.rows):

            self.problem += self.a[i] <= self.param_supply[i][1]
            self.problem += self.param_supply[i][0] <= self.a[i]

        for j in range(self.cols):

            self.problem += self.b[j] <= self.param_demand[j][1]
            self.problem += self.param_demand[j][0] <= self.b[j]

    def set_objective(self):
        self.problem += pulp.lpSum(self.param_costs[i][j][0] * self.x[i, j] for i in range(self.rows) for j in range(self.cols))
    
    def solve(self):
        self.problem.solve()
        
        return self.problem.status

    def get_solution(self):
        solution = {(i, j): [self.x[i, j].varValue, self.a[i].varValue, self.b[j].varValue] for i in range(self.rows) for j in range(self.cols)}
        return solution

class FuzzyDualTransportProblem:
    def __init__(self, costs, supply, demand):
        self.costs = costs  # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.rows = len(supply)
        self.cols = len(demand)
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMaximize)
        self.tfn = TrapezoidalFuzzyNumber()
    def parametrize_fuzzy_numbers(self, alpha):
        self.param_costs = np.zeros((self.rows, self.cols), dtype=object)
        self.param_supply = np.zeros((self.rows), dtype=object)
        self.param_demand = np.zeros((self.cols), dtype=object)
        self.param_costs_upper = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                self.param_costs[i][j] = self.tfn.parametric_form(self.costs[i][j], alpha)
                if i == self.rows - 1:
                    self.param_demand[j] = self.tfn.parametric_form(self.demand[j], alpha)
                    
            self.param_supply[i] = self.tfn.parametric_form(self.supply[i], alpha)
        
        for i in range(self.rows):
            for j in range(self.cols):
                self.param_costs_upper[i][j] = self.param_costs[i][j][1] 
        
    
    def create(self):    

        ampl = AMPL()
        
        # Definir el modelo
        ampl.eval("""

                set S;
                set D;

                var u{S} integer;
                var a{S} integer;
                var v{D} integer;
                var b{D} integer;
                
                param c_upper{S,D};
                param a_lower{S};
                param a_upper{S};
                param b_lower{D};
                param b_upper{D};

                maximize obj: sum{i in S} a[i]*u[i] + sum{j in D} b[j]*v[j];

                s.t. cost_constraint {i in S, j in D}:
                    u[i] + v[j] <= c_upper[i,j];

                s.t. supply_constraint:
                    sum {i in S} a[i] = sum {j in D} b[j];

                s.t. bounds_ai {i in S}:
                    a_lower[i] <= a[i] <= a_upper[i];

                s.t. bounds_bj {j in D}:
                    b_lower[j] <= b[j] <= b_upper[j];

                
                      


        """)

        ampl.set['S'] = range(1, self.rows + 1)
        ampl.set['D'] = range(1, self.cols + 1)
        for i in ampl.param['c_upper']:
            print(i)
        # Enviar los valores de la matriz de costos a AMPL
        ampl.param['c_upper'] = {(i+1, j+1): self.param_costs[i][j][1] for i in range(self.rows) for j in range(self.cols)}

        # Agregar valores de oferta y demanda a AMPL
        for i in range(self.rows):
            ampl.param['a_lower'][i+1] = self.param_supply[i][0]  # Límite inferior de la oferta
            ampl.param['a_upper'][i+1] = self.param_supply[i][1]  # Límite superior de la oferta
        
        for j in range(self.cols):
            ampl.param['b_lower'][j+1] = self.param_demand[j][0]  # Límite inferior de la demanda
            ampl.param['b_upper'][j+1] = self.param_demand[j][1]  # Límite superior de la demanda
        
        # Resolver el modelo usando Bonmin
        ampl.set_option("solver", "bonmin")
        ampl.solve()

        print(f"Costo total: {ampl.obj['obj'].value()}")

        # # Imprimir los resultados
        for i in range(self.rows):
            print(ampl.var['u'][i+1].value())
            print(ampl.var['a'][i+1].value())

        for j in range(self.cols):
            print(ampl.var['v'][j+1].value())
            print(ampl.var['b'][j+1].value())
        
        
        
   

class ApplicationPrimal():

    def __init__(self, costs, supply, demand, alpha):

        self.costs = costs
        self.supply = supply
        self.demand = demand
        self.rows = len(self.supply)
        self.cols = len(self.demand)
        self.alpha = alpha
        
    def solve_primal(self):
        solution_list = []
 
        tfn = TrapezoidalFuzzyNumber()
    
        # Crear el problema de transporte fuzzy
        fptp = FuzzyPrimalTransportProblem(self.costs, self.supply, self.demand)
        
        
        matrix_result = np.zeros((self.rows, self.cols), dtype=object)

        fptp.parametrize_fuzzy_numbers(0)
        fptp.create_variables()
        fptp.add_constraints()
        fptp.set_objective()
        status = fptp.solve()

        if status == pulp.LpStatusOptimal: 
            solution = fptp.get_solution()
            pprint(solution)
            for (i, j), value in solution.items():
                matrix_result[i][j] = value

            solution_list.append(matrix_result)
                
        pprint(solution_list)
        
        fptp.parametrize_fuzzy_numbers(self.alpha)
        fptp.create_variables()
        fptp.add_constraints()
        fptp.set_objective()
        status = fptp.solve()

        if status == pulp.LpStatusOptimal: 
            solution = fptp.get_solution()
            pprint(solution)
            for (i, j), value in solution.items():
                matrix_result[i][j] = value

            solution_list.append(matrix_result)
                
        pprint(solution_list)
        

class ApplicationDual():
        
    def __init__(self, costs, supply, demand, alpha):
        self.costs = costs
        self.supply = supply
        self.demand = demand
        self.rows = len(self.supply)
        self.cols = len(self.demand)
        self.alpha = alpha
        
    def solve_dual(self):
        solution_list = []
 
        tfn = TrapezoidalFuzzyNumber()
    
        # Crear el problema de transporte fuzzy
        fdtp = FuzzyDualTransportProblem(self.costs, self.supply, self.demand)
        
        
        matrix_result = np.zeros((self.rows, self.cols), dtype=object)

        fdtp.parametrize_fuzzy_numbers(1)
        fdtp.create()
        

        # if status == pulp.LpStatusOptimal: 
        #     solution = 
        #     pprint(solution)
        #     for (i, j), value in solution.items():
        #         matrix_result[i][j] = value

        #     solution_list.append(matrix_result)
                
        # pprint(solution_list)
        
        
    

if __name__ == "__main__":
    # Definir las entradas
    # costs = [[3, 2, 1], [4, 3, 2], [5, 4, 3]]  # Matriz de costos
    # supply = [(10, 12, 14), (15, 17, 19), (20, 22, 24)]  # Números fuzzy triangulares para la oferta
    # demand = [(12, 14, 16), (18, 20, 22), (15, 17, 19)]  # Números fuzzy triangulares para la demanda

    costs = [[(5, 7,  9, 11), (2, 4, 6, 8), (10, 12, 14, 16)], [(2, 3, 4, 5), (4, 5, 7, 8), (7, 8, 10, 12)]]  # Matriz de costos
    supply = [(13, 15, 17, 19), (8, 11, 13, 15)]  # Números fuzzy triangulares para la oferta
    demand = [(3, 5, 6, 8), (7, 9, 10, 11), (11, 12, 14, 15)]  # Números fuzzy triangulares para la demanda    

    #app_primal = ApplicationPrimal(costs, supply, demand)
    # app_primal.solve_primal()
    app_dual = ApplicationDual(costs, supply, demand)
    app_dual.solve_dual()
    # for i in range(len(supply)):
    #     for j in range(len(demand)):
    #         lower = float((solution_list[1][i][j] - solution_list[0][i][j])*0 + solution_list[0][i][j])
    #         upper = float((solution_list[3][i][j] - solution_list[2][i][j])*0 + solution_list[2][i][j])
    #         middle = float(max((solution_list[1][i][j] - solution_list[0][i][j])*1 + solution_list[0][i][j], (solution_list[3][i][j] - solution_list[2][i][j])*1 + solution_list[2][i][j]))
            
    #         matrix_result[i][j] = tfn.constant_multiplication(costs[i][j], (lower, middle, upper))
    # pprint(matrix_result)
    # optimus_solution = (0, 0, 0)

    # for i in range(len(supply)):
    #     for j in range(len(demand)):
    #         optimus_solution = tfn.sum_triangular_fuzzy_number(optimus_solution, matrix_result[i][j])

    
    
    # print(optimus_solution)