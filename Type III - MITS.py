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
    def __init__(self, costs_stage_one, costs_stage_two, supply, demand):
        self.costs_stage_one = costs_stage_one  # Matriz de costos
        self.costs_stage_two = costs_stage_two # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.products = len(self.costs_stage_one)
        self.sources = len(self.costs_stage_one[0])
        self.transhipments = len(self.costs_stage_two[0])
        self.destinations = len(self.costs_stage_two[0][0])
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.tfn = TrapezoidalFuzzyNumber()
    def parametrize_fuzzy_numbers(self, alpha):
        self.param_costs_stage_one = np.zeros((self.products, self.sources, self.transhipments), dtype=object)
        self.param_costs_stage_two = np.zeros((self.products, self.transhipments, self.destinations), dtype=object)
        self.param_supply = np.zeros((self.products, self.sources), dtype=object)
        self.param_demand = np.zeros((self.products, self.destinations), dtype=object)

        for p in range(self.products):
            for i in range(self.sources):
                for k in range(self.transhipments):
                    self.param_costs_stage_one[p][i][k] = self.tfn.parametric_form(self.costs_stage_one[p][i][k], alpha)
                    
                self.param_supply[p][i] = self.tfn.parametric_form(self.supply[p][i], alpha)    
        
        for p in range(self.products):
            for k in range(self.transhipments):
                for j in range(self.destinations):
                    self.param_costs_stage_two[p][k][j] = self.tfn.parametric_form(self.costs_stage_two[p][k][j], alpha)            
                    if k == self.transhipments - 1:
                        self.param_demand[p][j] = self.tfn.parametric_form(self.demand[p][j], alpha)
                    
            

        # pprint(self.param_supply)
        # pprint(self.param_demand)
    def create_variables(self):
        self.x = pulp.LpVariable.dicts("x", [(p, i, k) for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)], lowBound=0)
        self.y = pulp.LpVariable.dicts("y", [(p, k, j) for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)], lowBound=0)
        self.a = pulp.LpVariable.dicts("a", [(p, i) for p in range(self.products) for i in range(self.sources)])
        self.b = pulp.LpVariable.dicts("b", [(p, j) for p in range(self.products) for j in range(self.destinations)])
    
    def add_constraints(self):
        # Restricciones de oferta
        for p in range(self.products):
            for i in range(self.sources):
                
                self.problem += pulp.lpSum([self.x[p, i, k] for k in range(self.transhipments)]) == self.a[p,i]
            
        # Restricciones de demanda
        for p in range(self.products):
            for j in range(self.destinations):
                
                self.problem += pulp.lpSum([self.y[p, k, j] for k in range(self.transhipments)]) == self.b[p,j]

        for p in range(self.products):
            for k in range(self.transhipments):

                self.problem += pulp.lpSum([self.x[p, i, k] for i in range(self.sources)]) == pulp.lpSum([self.y[p, k, j] for j in range(self.destinations)])

        self.problem += pulp.lpSum([self.a[p,i] for p in range(self.products) for i in range(self.sources)]) == pulp.lpSum([self.b[p,j] for p in range(self.products) for j in range(self.destinations)])
    
        for p in range(self.products):
            for i in range(self.sources):

                self.problem += self.a[p,i] <= self.param_supply[p][i][1]
                self.problem += self.param_supply[p][i][0] <= self.a[p,i]

        for p in range(self.products):
            for j in range(self.destinations):

                self.problem += self.b[p,j] <= self.param_demand[p][j][1]
                self.problem += self.param_demand[p][j][0] <= self.b[p,j]

    def set_objective(self):
        self.problem += pulp.lpSum(self.param_costs_stage_one[p][i][k][0] * self.x[p, i, k] for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)) + pulp.lpSum(self.param_costs_stage_two[p][k][j][0] * self.y[p, k, j] for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations))
    
    def solve(self):
        self.problem.solve()
        
        return self.problem.status

    def get_solution(self):
        
        solution_stage_one = {(p, i, k): self.x[p, i, k].varValue for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)}
        solution_stage_two = {(p, k, j): self.y[p, k, j].varValue for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)}

        solution_supply = {(p, i): self.a[p, i].varValue for p in range(self.products) for i in range(self.sources)}
        solution_demand = {(p, j): self.b[p, j].varValue for p in range(self.products) for j in range(self.destinations)}

        pprint(solution_stage_one)
        pprint(solution_stage_two)

        return pulp.value(self.problem.objective), solution_stage_one, solution_stage_two, solution_supply, solution_demand

class FuzzyDualTransportProblem:
    def __init__(self, costs_stage_one, costs_stage_two, supply, demand):
        self.costs_stage_one = costs_stage_one  # Matriz de costos
        self.costs_stage_two = costs_stage_two # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.products = len(self.costs_stage_one)
        self.sources = len(self.costs_stage_one[0])
        self.transhipments = len(self.costs_stage_two[0])
        self.destinations = len(self.costs_stage_two[0][0])
        self.problem = pulp.LpProblem("Fuzzy_Transportation_Problem", pulp.LpMinimize)
        self.tfn = TrapezoidalFuzzyNumber()
    def parametrize_fuzzy_numbers(self, alpha):
        self.param_costs_stage_one = np.zeros((self.products, self.sources, self.transhipments), dtype=object)
        self.param_costs_stage_two = np.zeros((self.products, self.transhipments, self.destinations), dtype=object)
        self.param_supply = np.zeros((self.products, self.sources), dtype=object)
        self.param_demand = np.zeros((self.products, self.destinations), dtype=object)

        for p in range(self.products):
            for i in range(self.sources):
                for k in range(self.transhipments):
                    self.param_costs_stage_one[p][i][k] = self.tfn.parametric_form(self.costs_stage_one[p][i][k], alpha)
                    
                self.param_supply[p][i] = self.tfn.parametric_form(self.supply[p][i], alpha)    
        
        for p in range(self.products):
            for k in range(self.transhipments):
                for j in range(self.destinations):
                    self.param_costs_stage_two[p][k][j] = self.tfn.parametric_form(self.costs_stage_two[p][k][j], alpha)            
                    if k == self.transhipments - 1:
                        self.param_demand[p][j] = self.tfn.parametric_form(self.demand[p][j], alpha)
        
        # pprint(self.param_costs_stage_one)
        # pprint(self.param_costs_stage_two)

    def create(self):    

        ampl = AMPL()
        
        # Definir el modelo
        ampl.eval("""

                set P;
                set S;
                set T;
                set D;

                var u{p in P, i in S} integer;
                var a{p in P, i in S} integer;
                var v{p in P, j in D} integer;
                var b{p in P, j in D} integer;
                var w{p in P, k in T} integer >= 0;
                
                param c_upper_one{P,S,T};
                param c_upper_two{P,T,D};
                param a_lower{P,S};
                param a_upper{P,S};
                param b_lower{P,D};
                param b_upper{P,D};

                maximize obj: sum{p in P, i in S} a[p,i]*u[p,i] + sum{p in P, j in D} b[p,j]*v[p,j] + sum{p in P, k in T} 0*w[p,k];

                s.t. cost_constraint_one {p in P, i in S, k in T}:
                    u[p,i] + w[p,k] <= c_upper_one[p,i,k];
                  
                s.t. cost_constraint_two {p in P, k in T, j in D}:
                    w[p,k] + v[p,j] <= c_upper_two[p,k,j];
                  
                s.t. supply_constraint:
                    sum {p in P, i in S} a[p,i] = sum {p in P, j in D} b[p,j];

                s.t. bounds_ai {p in P, i in S}:
                    a_lower[p,i] <= a[p,i] <= a_upper[p,i];

                s.t. bounds_bj {p in P, j in D}:
                    b_lower[p,j] <= b[p,j] <= b_upper[p,j];

                
                      


        """)

        ampl.set['P'] = range(1, self.products + 1)
        ampl.set['S'] = range(1, self.sources + 1)
        ampl.set['T'] = range(1, self.transhipments + 1)
        ampl.set['D'] = range(1, self.destinations + 1)
        # for p,i in ampl.param['c_upper_one']:
        #     print(p,i)
        # Enviar los valores de la matriz de costos a AMPL
        ampl.param['c_upper_one'] = {(p+1, i+1, k+1): self.param_costs_stage_one[p][i][k][1] for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)}

        ampl.param['c_upper_two'] = {(p+1, k+1, j+1): self.param_costs_stage_two[p][k][j][1] for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)}
        for p in range(self.products):
            for i in range(self.sources):
                for k in range(self.transhipments):
                    print(ampl.param['c_upper_one'][p+1,i+1,k+1])
        # Agregar valores de oferta y demanda a AMPL
        for p in range(self.products):
            for i in range(self.sources):
                ampl.param['a_lower'][p+1,i+1] = self.param_supply[p][i][0]  # Límite inferior de la oferta
                ampl.param['a_upper'][p+1,i+1] = self.param_supply[p][i][1]  # Límite superior de la oferta
        
        for p in range(self.products):
            for j in range(self.destinations):
                ampl.param['b_lower'][p+1,j+1] = self.param_demand[p][j][0]  # Límite inferior de la demanda
                ampl.param['b_upper'][p+1,j+1] = self.param_demand[p][j][1]  # Límite superior de la demanda
        
        # Resolver el modelo usando Bonmin
        ampl.set_option("solver", "bonmin")
        ampl.solve()

        print(f"Costo total: {ampl.obj['obj'].value()}")
        supply_dual = np.zeros((self.products,self.sources))
        demand_dual = np.zeros((self.products,self.destinations))
        costs_stage_one_dual = np.zeros((self.products,self.sources, self.transhipments))
        costs_stage_two_dual = np.zeros((self.products,self.transhipments, self.destinations))
        # Imprimir los resultados
        for p in range(self.products):    
            for i in range(self.sources):
                supply_dual[p][i] = ampl.var['a'][p+1,i+1].value()
        
        for p in range(self.products):
            for k in range(self.transhipments):
                print(f'w_{p+1}{k+1}', ampl.var['w'][p+1,k+1].value())
                

        for p in range(self.products):
            for j in range(self.destinations):
                demand_dual[p][j] = ampl.var['b'][p+1,j+1].value()
        
        for p in range(self.products):
            for i in range(self.sources):
                for k in range(self.transhipments):
                    costs_stage_one_dual[p][i][k] = ampl.param['c_upper_one'][p+1,i+1,k+1]
    
        for p in range(self.products):
            for k in range(self.transhipments):
                for j in range(self.destinations):
                    costs_stage_two_dual[p][k][j] = ampl.param['c_upper_two'][p+1,k+1,j+1]
   
        return ampl.obj['obj'].value(), supply_dual, demand_dual, costs_stage_one_dual, costs_stage_two_dual
    
    def get_optimal_values(self, supply_dual, demand_dual, costs_stage_one_dual, costs_stage_two_dual):
        problem = pulp.LpProblem("Fuzzy_Transportation_Problem_Primal", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", [(p, i, k) for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)], lowBound=0)
        y = pulp.LpVariable.dicts("y", [(p, k, j) for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)], lowBound=0)
        
        problem += pulp.lpSum(costs_stage_one_dual[p][i][k] * x[p, i, k] for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)) + pulp.lpSum(costs_stage_two_dual[p][k][j] * y[p, k, j] for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations))
   
        # Restricciones de oferta
        for p in range(self.products):
            for i in range(self.sources):
                
                problem += pulp.lpSum([x[p, i, k] for k in range(self.transhipments)]) == supply_dual[p,i]
            
        # Restricciones de demanda
        for p in range(self.products):
            for j in range(self.destinations):
                
                problem += pulp.lpSum([y[p, k, j] for k in range(self.transhipments)]) == demand_dual[p,j]

        for p in range(self.products):
            for k in range(self.transhipments):

                problem += pulp.lpSum([x[p, i, k] for i in range(self.sources)]) == pulp.lpSum([y[p, k, j] for j in range(self.destinations)])

        problem += pulp.lpSum([supply_dual[p,i] for p in range(self.products) for i in range(self.sources)]) == pulp.lpSum([demand_dual[p,j] for p in range(self.products) for j in range(self.destinations)])

        problem.solve()


        solution_stage_one = {(p, i, k): x[p, i, k].varValue for p in range(self.products) for i in range(self.sources) for k in range(self.transhipments)}
        solution_stage_two = {(p, k, j): y[p, k, j].varValue for p in range(self.products) for k in range(self.transhipments) for j in range(self.destinations)}

        return solution_stage_one, solution_stage_two
class ApplicationPrimal():

    def __init__(self, costs_stage_one, costs_stage_two, supply, demand, alpha):

        
        self.costs_stage_one = costs_stage_one  # Matriz de costos
        self.costs_stage_two = costs_stage_two # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.products = len(self.costs_stage_one)
        self.sources = len(self.costs_stage_one[0])
        self.transhipments = len(self.costs_stage_two[0])
        self.destinations = len(self.costs_stage_two[0][0])
        self.alpha = alpha
        
    def solve_primal(self):
        solution_list = []
 
        tfn = TrapezoidalFuzzyNumber()
    
        # Crear el problema de transporte fuzzy
        fptp = FuzzyPrimalTransportProblem(self.costs_stage_one, self.costs_stage_two, self.supply, self.demand)
        
        
        matrix_result_stage_one = np.zeros((self.products, self.sources, self.transhipments), dtype=object)
        matrix_result_stage_two = np.zeros((self.products, self.transhipments, self.destinations), dtype=object)
        matrix_result_supply = np.zeros((self.products, self.sources))
        matrix_result_demand = np.zeros((self.products, self.destinations))

        fptp.parametrize_fuzzy_numbers(self.alpha)
        fptp.create_variables()
        fptp.add_constraints()
        fptp.set_objective()
        status = fptp.solve()

        if status == pulp.LpStatusOptimal: 
            optimus, solution_stage_one, solution_stage_two, solution_supply, solution_demand = fptp.get_solution()
            
            for (p, i, k), value in solution_stage_one.items():
                matrix_result_stage_one[p][i][k] = value

            for (p, k, j), value in solution_stage_two.items():
                matrix_result_stage_two[p][k][j] = value

            for (p, i), value in solution_supply.items():
                matrix_result_supply[p][i] = value

            for (p, j), value in solution_demand.items():
                matrix_result_demand[p][j] = value

            # solution_list = [matrix_result_stage_one, matrix_result_stage_two, matrix_result_supply, matrix_result_demand]
                
        return optimus, matrix_result_stage_one, matrix_result_stage_two, matrix_result_supply, matrix_result_demand
        
        
        

class ApplicationDual():
        
    def __init__(self, costs_stage_one, costs_stage_two, supply, demand, alpha):
        
        self.costs_stage_one = costs_stage_one  # Matriz de costos
        self.costs_stage_two = costs_stage_two # Matriz de costos
        self.supply = supply  # Vector de oferta (fuzzy)
        self.demand = demand  # Vector de demanda (fuzzy)
        self.products = len(self.costs_stage_one)
        self.sources = len(self.costs_stage_one[0])
        self.transhipments = len(self.costs_stage_two[0])
        self.destinations = len(self.costs_stage_two[0][0])
        self.alpha = alpha
        
    def solve_dual(self):
        solution_list = []
 
        tfn = TrapezoidalFuzzyNumber()
    
        # Crear el problema de transporte fuzzy
        fdtp = FuzzyDualTransportProblem(self.costs_stage_one, self.costs_stage_two, self.supply, self.demand)
        
        
        # matrix_result_stage_one = np.zeros((self.rows, self.cols), dtype=object)

        fdtp.parametrize_fuzzy_numbers(self.alpha)
        
        optimus, supply_dual, demand_dual, costs_stage_one_dual, costs_stage_two_dual = fdtp.create()
        
        matrix_result_stage_one_dual = np.zeros((self.products,self.sources, self.transhipments))
        matrix_result_stage_two_dual = np.zeros((self.products,self.transhipments, self.destinations))

        solution_stage_one, solution_stage_two = fdtp.get_optimal_values(supply_dual, demand_dual, costs_stage_one_dual, costs_stage_two_dual)
            
        for (p, i, k), value in solution_stage_one.items():
            matrix_result_stage_one_dual[p][i][k] = value

        for (p, k, j), value in solution_stage_two.items():
            matrix_result_stage_two_dual[p][k][j] = value
        
        return optimus, matrix_result_stage_one_dual, matrix_result_stage_two_dual, supply_dual, demand_dual
        # if status == pulp.LpStatusOptimal: 
        #     solution = 
        #     pprint(solution)
        #     for (i, j), value in solution.items():
        #         matrix_result[i][j] = value

        #     solution_list.append(matrix_result)
                
        # pprint(solution_list)
        
class FuzzyChart():
    def show_chart(self, list_optimus_primal, list_optimus_dual, list_alpha_primal, list_alpha_dual):
        
        # Configurar matplotlib para usar LaTeX
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')  # Usar Computer Modern, que es la fuente por defecto de LaTeX

        x_increasing = list_optimus_primal  # Eje x
        y_increasing = list_alpha_primal  # Eje y

        # Datos constantes
        x_constant = np.linspace(list_optimus_primal[len(list_optimus_primal)-1], list_optimus_dual[len(list_optimus_dual)-1], 20) # Eje x
        y_constant = np.linspace(1,1,20)  # Valor constante en y

        # Datos decrecientes
        list_optimus_dual.reverse()
        list_alpha_dual.reverse()

        x_decreasing = list_optimus_dual  # Eje x
        y_decreasing = list_alpha_dual  # Eje y

        # Definir el rango de la función fuera de x
        x_out = list(range(int(list_optimus_primal[0])-6000, int(list_optimus_dual[len(list_optimus_dual)-1])+6000))  # Ejemplo del rango completo en eje x
        y_out = [0 if x < list_optimus_primal[0] or x > list_optimus_dual[len(list_optimus_dual)-1] else None for x in x_out]  # Valores 0 fuera del rango de x

        # Graficar por partes
        plt.plot(x_increasing, y_increasing, label='Valores optimos primales', color='green')
        plt.plot(x_constant, y_constant, label='Constante', color='orange')
        plt.plot(x_decreasing, y_decreasing, label='Valores óptimos duales', color='red')

        # Graficar los valores fuera del rango
        plt.plot(x_out, y_out, label='No pertenencia', color='blue')

        # Añadir etiquetas y leyenda
        plt.xlabel(r'Unidades monetarias (\$)')
        plt.ylabel(r'Grado de pertenencia $\alpha$')
        plt.title('Número LR flat óptimo')
        plt.legend()

        # Mostrar la gráfica
        plt.grid(True)
        plt.show()
class Network():

    def show_network(self, allocation_stage_one, allocation_stage_two, supply, demand, type_problem, p, alpha):
        # Crear un grafo vacío
        G = nx.DiGraph()  # Grafo dirigido, porque hay un flujo de origen -> transbordo -> destino
        # Configurar matplotlib para usar LaTeX
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')  # Usar Computer Modern, que es la fuente por defecto de LaTeX

        # Añadir nodos de orígenes, transbordos y destinos
        sources_list = [f"Source {i+1}: {supply[p][i]}" for i in range(len(allocation_stage_one))]
        transhipments_list = [f"Transhipment {i+1}" for i in range(len(allocation_stage_two))]
        destinations_list = [f"Destination {i+1}: {demand[p][i]}" for i in range(len(allocation_stage_two[0]))]

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
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=4000)

        # Dibujar etiquetas
        nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold')

        # Dibujar aristas con diferentes colores
        nx.draw_networkx_edges(G, pos, edge_color='red', arrows=True, arrowstyle='->', arrowsize=20)

        # Mostrar el grafo
        plt.title(f'Red óptima {type_problem} de producto {p+1} para alfa {alpha}')
        plt.show()
    

if __name__ == "__main__":
    # Definir las entradas
    # costs = [[3, 2, 1], [4, 3, 2], [5, 4, 3]]  # Matriz de costos
    # supply = [(10, 12, 14), (15, 17, 19), (20, 22, 24)]  # Números fuzzy triangulares para la oferta
    # demand = [(12, 14, 16), (18, 20, 22), (15, 17, 19)]  # Números fuzzy triangulares para la demanda
    tfn = TrapezoidalFuzzyNumber()
    

    # supply = [
    #             [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)],
    #             [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)]
    #         ]  # Números fuzzy triangulares para la oferta
    # demand = [
    #             [tfn.generate_trapezoidal_fuzzy_number(10, 60), tfn.generate_trapezoidal_fuzzy_number(10, 60)],
    #             [tfn.generate_trapezoidal_fuzzy_number(10, 60), tfn.generate_trapezoidal_fuzzy_number(10, 60)]
    #         ]  # Números fuzzy triangulares para la demanda    
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
    #     new_tfn = sum_supply_by_row[i][0] - sum_demand_by_row[i][0], sum_supply_by_row[i][1] - sum_demand_by_row[i][1], sum_supply_by_row[i][2] - sum_demand_by_row[i][2], sum_supply_by_row[i][3] - sum_demand_by_row[i][3]
    #     demand[i] = demand[i] + [new_tfn]


    # costs_stage_one = [
    #                     [
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)], 
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)]
    #                     ],
    #                     [
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)], 
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)]
    #                     ]
    #                 ]  # Matriz de costos
    # costs_stage_two = [
    #                     [
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)], 
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)]
    #                     ],
    #                     [
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)], 
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90), tfn.generate_trapezoidal_fuzzy_number(10, 90)]
    #                     ]
    #                 ]  # Matriz de costos
    
    supply = [
                [(28, 38, 48, 56), (36, 46, 57, 68)], 
                [(34, 45, 58, 66), (40, 52, 60, 68)]
            ]
    demand = [
                [(28, 33, 38, 43), (32, 38, 44, 49), (4, 13, 23, 32)],
                [(26, 31, 37, 42), (21, 26, 31, 41), (27, 40, 50, 51)]
            ]
    costs_stage_one = [
                        [
                            [(48, 56, 64, 72), (43, 51, 59, 72)], [(42, 50, 59, 67), (37, 69, 77, 85)]
                        ],
                        [
                            [(38, 46, 54, 75), (43, 57, 65, 73)], [(37, 45, 53, 66), (35, 59, 67, 75)]
                        ]
                    ]
    costs_stage_two = [
                        [
                            [(43, 55, 70, 78), (45, 53, 61, 88), (37, 45, 53, 61)],
                            [(30, 45, 60, 68), (22, 34, 43, 51), (47, 58, 66, 74)]
                        ],
                        [
                            [(30, 44, 52, 89), (33, 43, 64, 84), (26, 52, 60, 68)],
                            [(37, 45, 55, 70), (30, 45, 56, 64), (31, 42, 50, 58)]
                        ]
                    ]

    # pprint(supply)
    # pprint(demand)
    # pprint(costs_stage_one)
    # pprint(costs_stage_two)

    list_optimus_primal = []
    list_optimus_dual = []
    list_alpha_primal = []
    
    net = Network()

    for i in range(0, 11, 2):
        
        app_primal = ApplicationPrimal(costs_stage_one, costs_stage_two, supply, demand, i/10)
        optimus_primal, matrix_result_stage_one_primal, matrix_result_stage_two_primal, supply_result_primal, demand_result_primal = app_primal.solve_primal()
        
        list_optimus_primal.append(optimus_primal)

        app_dual = ApplicationDual(costs_stage_one, costs_stage_two, supply, demand, i/10)
        optimus_dual, matrix_result_stage_one_dual, matrix_result_stage_two_dual, supply_result_dual, demand_result_dual = app_dual.solve_dual()

        list_optimus_dual.append(optimus_dual)

        list_alpha_primal.append(i/10)

        
            
        for p in range(len(costs_stage_one)):
            if i/10 == 0 or i/10 == 1:
                pprint(matrix_result_stage_one_primal)
                pprint(matrix_result_stage_two_primal)
                pprint(matrix_result_stage_one_dual)
                pprint(matrix_result_stage_two_dual)
                net.show_network(matrix_result_stage_one_primal[p], matrix_result_stage_two_primal[p], supply_result_primal, demand_result_primal, 'primal', p, i/10)
                
                net.show_network(matrix_result_stage_one_dual[p], matrix_result_stage_two_dual[p], supply_result_dual, demand_result_dual, 'dual', p, i/10)

                
            

    list_alpha_dual = list_alpha_primal.copy()

    fuzzy_chart = FuzzyChart()
    fuzzy_chart.show_chart(list_optimus_primal, list_optimus_dual, list_alpha_primal, list_alpha_dual)

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