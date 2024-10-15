import numpy as np
from pprint import pprint
import random
import matplotlib.pyplot as plt
import networkx as nx



class TrapezoidalFuzzyNumber():


    def sum_trapezoidal_fuzzy_numbers(self, tuple1, tuple2):
        """

        :param tuple1: Tupla de número trapezoidal difuso y valor de pertenencia (el número trapezoidal debe estar compuesto por 4 enteros).
        :param tuple2: Tupla de número trapezoidal difuso y valor de pertenencia (el número trapezoidal debe estar compuesto por 4 enteros).
        :return: Tupla de número trapezoidal difuso sumado y valor de pertenencia mínimo (el número trapezoidal debe estar compuesto por 4 enteros).
        """
        trapezoidal_number1, membership1 = tuple1        
        trapezoidal_number2, membership2 = tuple2
        # Sumar los primeros 4 elementos del vector 1 con los primeros 4 elementos del vector 2 en orden invertido
        summed_trapezoidal_number = [trapezoidal_number1[i] + trapezoidal_number2[i] for i in range(len(trapezoidal_number1))]

        # Elegir el mínimo entre las funciones de pertenencia
        min_membership = min(membership1, membership2)

        # Crear el vector de resultado de la suma
        final_tuple = (summed_trapezoidal_number, min_membership)

        return final_tuple
    
    def rest_trapezoidal_fuzzy_numbers(self, tuple1, tuple2):
        """

        :param tuple1: Tupla de número trapezoidal difuso y valor de pertenencia (el número trapezoidal debe estar compuesto por 4 enteros y la pertenencia debe estar entre 0 y 1).
        :param tuple2: Tupla de número trapezoidal difuso y valor de pertenencia (el número trapezoidal debe estar compuesto por 4 enteros y la pertenencia debe estar entre 0 y 1).
        :return: Tupla de número trapezoidal difuso restado y valor de pertenencia mínimo (el número trapezoidal debe estar compuesto por 4 enteros y la pertenencia debe estar entre 0 y 1).
        """

        trapezoidal_number1, membership1 = tuple1        
        trapezoidal_number2, membership2 = tuple2
        # Sumar los primeros 4 elementos del vector 1 con los primeros 4 elementos del vector 2 en orden invertido
        rested_trapezoidal_number = [trapezoidal_number1[i] - trapezoidal_number2[len(trapezoidal_number1)-1-i] for i in range(len(trapezoidal_number1))]

        # Elegir el mínimo entre las funciones de pertenencia
        min_membership = min(membership1, membership2)

        # Crear el vector de resultado de la suma
        final_tuple = (rested_trapezoidal_number, min_membership)

        return final_tuple


    def constant_multiplication(self, constant, tuple):
        trapezoidal_fuzzy_number, membership = tuple

        trapezoidal_fuzzy_number = [constant*i for i in trapezoidal_fuzzy_number]

        return (trapezoidal_fuzzy_number, membership)

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

        alpha = round(random.uniform(0.2, 0.8),1)

        return (base_numbers, alpha)

class KaurAndKumarMethod:




    def __init__(self, supply, demand, costs):
        self.supply = supply.copy()
        self.demand = demand.copy()
        self.costs = costs
        self.allocation = np.zeros((len(supply), len(demand)))
        self.tfn = TrapezoidalFuzzyNumber()
    

    def north_west_corner_method(self):
        i, j = 0, 0  # índice de la oferta y la demanda
        while i < len(self.supply) and j < len(self.demand):
            min_value = min(self.supply[i], self.demand[j])
            self.allocation[i][j] = min_value
            self.supply[i] -= min_value
            self.demand[j] -= min_value

            if self.supply[i] == 0:
                i += 1
            if self.demand[j] == 0:
                j += 1

        print("Asignación inicial (método de la esquina noroeste):")
        pprint(self.allocation)

    
    def least_cost_method(self):
        
        weighted_costs = np.zeros((len(self.supply), len(self.demand)))

        for i in range(len(self.supply)):
            for j in range(len(self.demand)):
                weighted_costs[i][j] = sum(costs[i][j][0])/len(costs[i][j][0])
        
        
        # Bucle mientras haya oferta o demanda pendiente
        while np.any(self.supply) and np.any(self.demand):
            
            # Encontrar el índice de la celda con el costo más bajo
            min_cost_index = np.unravel_index(np.argmin(weighted_costs, axis=None), weighted_costs.shape)
            i, j = min_cost_index
            
            # Determinar la cantidad a asignar (mínimo entre oferta y demanda disponible)
            allocation_min = min(self.supply[i], self.demand[j])
            self.allocation[i][j] = allocation_min
            
            # Actualizar la oferta y la demanda
            self.supply[i] -= allocation_min
            self.demand[j] -= allocation_min
            
            # Si la oferta se ha agotado, eliminar la fila correspondiente
            if self.supply[i] == 0:
                weighted_costs[i, :] = np.inf
                
            
            # Si la demanda se ha agotado, eliminar la columna correspondiente
            if self.demand[j] == 0:
                weighted_costs[:, j] = np.inf
                
        print("Asignación inicial (método de Costo Mínimo):")
        pprint(self.allocation)

    def solve_fuzzy_dual(self):
        rows, cols = self.allocation.shape
        ui = np.full(rows, None)  # Variables fuzzy para las filas
        vj = np.full(cols, None)  # Variables fuzzy para las columnas
        
        duals = []

        
        
        for i in range(rows):
            for j in range(cols):
                if self.allocation[i][j]>0:
                    duals.append(i)
        if rows + cols - len(duals) > 1:
            unique_values, counts = np.unique(duals, return_counts=True)
            repeated = unique_values[counts > 1]
            for i in range(len(ui)):
                if i not in repeated:
                    ui[i] = ([0, 0, 0, 0], 1)  # Asignamos las variables unicas basicas como 0
            
            
        else:
            ui[0] = ([0, 0, 0, 0], 1)

        while None in ui or None in vj:
            for i in range(rows):
                for j in range(cols):
                    if self.allocation[i][j] > 0:  # Solo celdas básicas
                        if ui[i] is not None and vj[j] is None:
                            vj[j] = self.tfn.rest_trapezoidal_fuzzy_numbers(self.costs[i][j], ui[i])
                        elif vj[j] is not None and ui[i] is None:
                            ui[i] = self.tfn.rest_trapezoidal_fuzzy_numbers(self.costs[i][j], vj[j])

        return ui, vj

    def calculate_deltas(self, ui, vj):
        rows, cols = self.allocation.shape
        deltas = np.zeros((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                if self.allocation[i][j] == 0:  # Solo para celdas no básicas
                    sum_ui_vj = self.tfn.sum_trapezoidal_fuzzy_numbers(ui[i], vj[j])
                    deltas[i][j] = self.tfn.rest_trapezoidal_fuzzy_numbers(self.costs[i][j], sum_ui_vj)
                    
                else: 
                    deltas[i][j] = ([0, 0, 0, 0], 1)

        return deltas

    def find_most_negative_delta(self, deltas):
        rows, cols = self.allocation.shape
        min_deltas = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                delta_temp = deltas[i][j][0]
                min_deltas[i][j] = sum(delta_temp)/len(delta_temp)

        min_value = np.min(min_deltas)
        print(min_value)
        if min_value >= 0:
            return None  # La solución es óptima
        return np.unravel_index(np.argmin(min_deltas), min_deltas.shape)  # Retorna la celda con el delta más negativo

    def find_closed_cycle(self, entering_cell):
        rows, cols = self.allocation.shape
        i, j = entering_cell
        path = [(i, j)]  # Inicializar el camino con la celda entrante
        visited = set()  # Usar un conjunto para rastrear celdas visitadas
        visited.add((i, j))

        while True:
            if len(path) % 2 == 1:  # Movimientos horizontales (filas fijas, columnas cambian)
                found = False
                for col in range(cols):
                    if col != j and self.allocation[i][col] > 0:
                        if (i, col) not in visited:
                            j = col
                            path.append((i, j))
                            visited.add((i, j))
                            found = True
                            
                            break
                        

                            
                        
                if not found:
                    if path:
                        
                        visited.add((i, j))
                        path.pop()  # Retrocede a la celda anterior
                        i, j = path[-1]
                        

            else:  # Movimientos verticales (columnas fijas, filas cambian)
                found = False
                for row in range(rows):
                    if row != i and self.allocation[row][j] > 0:
                        if (row, j) not in visited:
                            i = row
                            path.append((i, j))
                            visited.add((i, j))
                            found = True
                            break

                        
                if not found:
                    for row in range(rows):
                        if row != i and self.allocation[row][j] == 0:
                            if (row, j) not in visited:
                                i = row
                                path.append((row, j))
                                visited.add((row, j))
                                found = True
                                break

                            elif (row, j) == path[0]:
                                i = row
                                break
                            else:
                                path.pop()
                                i, j = path[-1]
                                break
                               

            # Si has vuelto a la celda de inicio, cierra el ciclo
            if path[0] == (i, j):
                break

        return path  # Asegurar que es un ciclo válido
    def update_allocation(self, path):
        
        min_value = min(self.allocation[i][j] for i, j in path[1::2])
        for k, (i, j) in enumerate(path):
            if k % 2 == 0:
                self.allocation[i][j] += min_value
            else:
                self.allocation[i][j] -= min_value

    def solve(self):
        #self.north_west_corner_method()
        self.least_cost_method()

        while True:
            ui, vj = self.solve_fuzzy_dual()
            print("Variables fuzzy duales (ui, vj):", ui, vj)
            
            deltas = self.calculate_deltas(ui, vj)
            print("Deltas para celdas no básicas:")
            pprint(deltas)
            
            entering_cell = self.find_most_negative_delta(deltas)
            
            if entering_cell is None:
                print("Solución óptima encontrada.")

                rows, cols = self.allocation.shape
                matrix_result = np.zeros((rows, cols), dtype=object)

                for i in range(rows):
                    for j in range(cols):

                        matrix_result[i][j] = self.tfn.constant_multiplication(self.allocation[i][j], self.costs[i][j])

                optimus_solution = ([0, 0, 0, 0], 1)

                for i in range(rows):
                    for j in range(cols):
                        optimus_solution = self.tfn.sum_trapezoidal_fuzzy_numbers(optimus_solution, matrix_result[i][j])

                return optimus_solution
                break
            else:
                print(f"Seleccionando la celda {entering_cell} para entrar.")
                
                path = self.find_closed_cycle(entering_cell)
                print(f"Ciclo cerrado: {path}")
                
                self.update_allocation(path)
                print("Nueva asignación después de ajustar el ciclo:")
                pprint(self.allocation)


class EbrahimnejadMethod:




    def __init__(self, supply, demand, costs_stage_one, costs_stage_two):
        self.supply = supply.copy()
        self.demand = demand.copy()
        self.costs_stage_one = costs_stage_one.copy()
        self.costs_stage_two = costs_stage_two.copy()
        self.allocation_stage_one = np.zeros((len(self.costs_stage_one), len(self.costs_stage_one[0])))
        self.allocation_stage_two = np.zeros((len(self.costs_stage_two), len(self.costs_stage_two[0])))
        self.tfn = TrapezoidalFuzzyNumber()

    def adjust_costs(self):

        adjusted_costs_stage_one = np.zeros((len(self.costs_stage_one), len(self.costs_stage_one[0])))
        adjusted_costs_stage_two = np.zeros((len(self.costs_stage_two), len(self.costs_stage_two[0])))
        min_membership_stage_one = 1
        min_membership_stage_two = 1

        for i in range(len(adjusted_costs_stage_one)):
            for j in range(len(adjusted_costs_stage_one[0])):
                if self.costs_stage_one[i][j][1] < min_membership_stage_one:
                    min_membership_stage_one = self.costs_stage_one[i][j][1]
                    
        for i in range(len(adjusted_costs_stage_two)):
            for j in range(len(adjusted_costs_stage_two[0])):
                if self.costs_stage_two[i][j][1] < min_membership_stage_two:
                    min_membership_stage_two = self.costs_stage_two[i][j][1]
                    
        for i in range(len(adjusted_costs_stage_one)):
            for j in range(len(adjusted_costs_stage_one[0])):
                adjusted_costs_stage_one[i][j] = min_membership_stage_one*(sum(self.costs_stage_one[i][j][0])/4)
                
        for i in range(len(adjusted_costs_stage_two)):
            for j in range(len(adjusted_costs_stage_two[0])):
                adjusted_costs_stage_two[i][j] = min_membership_stage_two*(sum(self.costs_stage_two[i][j][0])/4)
        pprint(adjusted_costs_stage_one)
        pprint(adjusted_costs_stage_two)
        return adjusted_costs_stage_one, adjusted_costs_stage_two

    # def north_west_corner_method(self):
    #     i, j = 0, 0  # índice de la oferta y la demanda
    #     while i < len(self.supply) and j < len(self.demand):
    #         min_value = min(self.supply[i], self.demand[j])
    #         self.allocation[i][j] = min_value
    #         self.supply[i] -= min_value
    #         self.demand[j] -= min_value

    #         if self.supply[i] == 0:
    #             i += 1
    #         if self.demand[j] == 0:
    #             j += 1

    #     print("Asignación inicial (método de la Esquina Noroccidental):")
    #     pprint(self.allocation)

    def least_cost_adjusted_method(self):
        
        adjusted_costs_stage_one, adjusted_costs_stage_two = self.adjust_costs()
        
        # Bucle mientras haya oferta o demanda pendiente
        while np.any(self.supply) and np.any(self.demand):
            
            # Encontrar el índice de la celda con el costo más bajo
            min_cost_index_stage_one = np.unravel_index(np.argmin(adjusted_costs_stage_one, axis=None), adjusted_costs_stage_one.shape)
            min_cost_index_stage_two = np.unravel_index(np.argmin(adjusted_costs_stage_two, axis=None), adjusted_costs_stage_two.shape)
            
            if adjusted_costs_stage_one[min_cost_index_stage_one[0]][min_cost_index_stage_one[1]] <= adjusted_costs_stage_two[min_cost_index_stage_two[0]][min_cost_index_stage_two[1]]:
                min_cost_index = min_cost_index_stage_one
                i, j = min_cost_index
                self.allocation_stage_one[i][j] += self.supply[i]
                adjusted_costs_stage_one[i, :] = np.inf
                
                min_cost_row_index = np.argmin(adjusted_costs_stage_two[j])

                while self.supply[i] != 0:
                    

                    # Determinar la cantidad a asignar (mínimo entre oferta y demanda disponible)
                    allocation_min = min(self.supply[i], self.demand[min_cost_row_index])
                    self.allocation_stage_two[j][min_cost_row_index] += allocation_min

                    # Actualizar la oferta y la demanda
                    self.supply[i] -= allocation_min
                    self.demand[min_cost_row_index] -= allocation_min

                    if self.demand[min_cost_row_index] == 0:
                        adjusted_costs_stage_two[j][min_cost_row_index] = np.inf
                        min_cost_row_index = np.argmin(adjusted_costs_stage_two[j])


            else:
                min_cost_index = min_cost_index_stage_two
                i, j = min_cost_index
                self.allocation_stage_two[i][j] += self.demand[j]
                adjusted_costs_stage_two[:, j] = np.inf
                
                min_cost_column_index = np.argmin(adjusted_costs_stage_one[:, i])

                while self.demand[j] != 0:
                    
                    # Determinar la cantidad a asignar (mínimo entre oferta y demanda disponible)
                    allocation_min = min(self.supply[min_cost_column_index], self.demand[j])
                    self.allocation_stage_one[min_cost_column_index][i] += allocation_min

                    # Actualizar la oferta y la demanda
                    self.supply[min_cost_column_index] -= allocation_min
                    self.demand[j] -= allocation_min

                    if self.supply[min_cost_column_index] == 0:
                        adjusted_costs_stage_one[min_cost_column_index][i] = np.inf
                        min_cost_column_index = np.argmin(adjusted_costs_stage_one[:, i])
            
                
        print("Asignación inicial (método de Costo Mínimo):")
        pprint(self.allocation_stage_one)
        pprint(self.allocation_stage_two)

    def solve_fuzzy_dual(self):
        rows_stage_one, cols_stage_one = self.allocation_stage_one.shape
        rows_stage_two, cols_stage_two = self.allocation_stage_two.shape
        ui_stage_one = np.full(rows_stage_one, None)  # Variables fuzzy para las filas
        vj_stage_one = np.full(cols_stage_one, None)  # Variables fuzzy para las columnas
        ui_stage_two = np.full(rows_stage_two, None)  # Variables fuzzy para las filas
        vj_stage_two = np.full(cols_stage_two, None)  # Variables fuzzy para las columnas
        adjusted_costs_stage_one, adjusted_costs_stage_two = self.adjust_costs()
        duals_stage_one = []
        duals_stage_two = []
        is_ui_stage_one = False
        is_vj_stage_one = False
        is_ui_stage_two = False
        is_vj_stage_two = False
        
        for i, row in enumerate(self.allocation_stage_one):
            if all(x == 0 for x in row):
                is_ui_stage_one, index_u_stage_one = True, i
        
        # Check for columns of zeros
        for j in range(len(self.allocation_stage_one[0])):
            if all(self.allocation_stage_one[i][j] == 0 for i in range(len(self.allocation_stage_one))):
                is_vj_stage_one, index_v_stage_one = True, j
        
        if is_ui_stage_one:
            ui_stage_one[index_u_stage_one] = 0

        if is_vj_stage_one:
            vj_stage_one[index_v_stage_one] = 0

        for i, row in enumerate(self.allocation_stage_two):
            if all(x == 0 for x in row):
                is_ui_stage_two, index_u_stage_two = True, i
        
        # Check for columns of zeros
        for j in range(len(self.allocation_stage_two[0])):
            if all(self.allocation_stage_one[i][j] == 0 for i in range(len(self.allocation_stage_one))):
                is_vj_stage_two, index_v_stage_two = True, j
        
        if is_ui_stage_two:
            ui_stage_two[index_u_stage_two] = 0

        if is_vj_stage_two:
            vj_stage_two[index_v_stage_two] = 0

        for i in range(rows_stage_one):
            for j in range(cols_stage_one):
                if self.allocation_stage_one[i][j]>0:
                    duals_stage_one.append((i, j))
        relations_stage_one = []
        is_row_stage_one = []
        for i in range(len(duals_stage_one)-1):
            for j in range(i+1, len(duals_stage_one)):
                if duals_stage_one[i][0] == duals_stage_one[j][0] or duals_stage_one[i][1] == duals_stage_one[j][1]:
                    if duals_stage_one[i][0] == duals_stage_one[j][0]:
                        row_stage_one = 0
                    else:
                        row_stage_one = 1

                    relations_stage_one.append((i,j))
                    is_row_stage_one.append(row_stage_one)


        relations_stage_one = list(set(relations_stage_one))
                    
        conected_stage_one = []
        disconected_stage_one = []

        
        if  len(duals_stage_one) - len(relations_stage_one) > 1:
            
            if not relations_stage_one:
                for i in range(len(ui_stage_one)):
                
                    ui_stage_one[i] = 0
            else:
                for relation in relations_stage_one:
                    conected_stage_one.append(duals_stage_one[relation[0]][0])
                    conected_stage_one.append(duals_stage_one[relation[0]][1])
                    conected_stage_one.append(duals_stage_one[relation[1]][0])
                    conected_stage_one.append(duals_stage_one[relation[1]][1])

                for i in range(rows_stage_one):
                    if i not in conected_stage_one:
                        disconected_stage_one.append(i)
                # print(duals_stage_one)
                # print(relations_stage_one)
                # print(conected_stage_one, disconected_stage_one)

                if disconected_stage_one:
                    for i in disconected_stage_one:
                        ui_stage_one[i] = 0
                        ui_stage_one[0] = 0
        else:
            ui_stage_one[0] = 0

        while None in ui_stage_one or None in vj_stage_one:
            for i in range(rows_stage_one):
                for j in range(cols_stage_one):
                    if self.allocation_stage_one[i][j] > 0:  # Solo celdas básicas
                        if ui_stage_one[i] is not None and vj_stage_one[j] is None:
                            vj_stage_one[j] = adjusted_costs_stage_one[i][j] - ui_stage_one[i]
                        elif vj_stage_one[j] is not None and ui_stage_one[i] is None:
                            ui_stage_one[i] = adjusted_costs_stage_one[i][j] - vj_stage_one[j]
            

        
        for i in range(rows_stage_two):
            for j in range(cols_stage_two):
                if self.allocation_stage_two[i][j]>0:
                    duals_stage_two.append((i, j))
        relations_stage_two = []
        is_row_stage_two = []

        for i in range(len(duals_stage_two)-1):
            for j in range(i+1, len(duals_stage_two)):
                if duals_stage_two[i][0] == duals_stage_two[j][0] or duals_stage_two[i][1] == duals_stage_two[j][1]:
                    if duals_stage_two[i][0] == duals_stage_two[j][0]:
                        row_stage_two = 0
                    else:
                        row_stage_two = 1

                    relations_stage_two.append((i,j))
                    is_row_stage_two.append(row_stage_two)


        relations_stage_two = list(set(relations_stage_two))
                    

        conected_stage_two = []
        disconected_stage_two = []

        if  len(duals_stage_two) - len(relations_stage_two) > 1:
            
            if not relations_stage_two:
                for i in range(len(ui_stage_two)):
                
                    ui_stage_two[i] = 0
            else:
                for relation in relations_stage_two:
                    conected_stage_two.append(duals_stage_two[relation[0]][0])
                    conected_stage_two.append(duals_stage_two[relation[0]][1])
                    conected_stage_two.append(duals_stage_two[relation[1]][0])
                    conected_stage_two.append(duals_stage_two[relation[1]][1])

                for i in range(rows_stage_two):
                    if i not in conected_stage_two:
                        disconected_stage_two.append(i)
                if disconected_stage_two:
                    for i in disconected_stage_two:
                        ui_stage_two[i] = 0
                        ui_stage_two[0] = 0
        else:
            ui_stage_two[0] = 0
            
        
        while None in ui_stage_two or None in vj_stage_two:
            for i in range(rows_stage_two):
                for j in range(cols_stage_two):
                    if self.allocation_stage_two[i][j] > 0:  # Solo celdas básicas
                        if ui_stage_two[i] is not None and vj_stage_two[j] is None:
                            vj_stage_two[j] = adjusted_costs_stage_two[i][j] - ui_stage_two[i]
                        elif vj_stage_two[j] is not None and ui_stage_two[i] is None:
                            ui_stage_two[i] = adjusted_costs_stage_two[i][j] - vj_stage_two[j]
            
      
        
        return ui_stage_one, vj_stage_one, ui_stage_two, vj_stage_two

    def calculate_deltas(self, ui, vj, allocation):
        rows, cols = allocation.shape
        deltas = np.zeros((rows, cols))
        adjusted_costs_stage_one, adjusted_costs_stage_two = self.adjust_costs()

        if allocation.shape == adjusted_costs_stage_one.shape:
            adjusted_costs = adjusted_costs_stage_one
        else:
            adjusted_costs = adjusted_costs_stage_two

        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] == 0:  # Solo para celdas no básicas
                    
                    deltas[i][j] = adjusted_costs[i][j] - (ui[i] + vj[j])

        return deltas

    def find_most_negative_delta(self, deltas):
        min_value = np.min(deltas)
        if min_value >= 0:
            return None  # La solución es óptima
        return np.unravel_index(np.argmin(deltas), deltas.shape)  # Retorna la celda con el delta más negativo

    def find_closed_cycle(self, entering_cell, allocation):
        
        rows, cols = allocation.shape
    
        i, j = entering_cell
        path = [(i, j)]  # Inicializar el camino con la celda entrante
        visited = set()  # Usar un conjunto para rastrear celdas visitadas
        visited.add((i, j))

        while True:
            if len(path) % 2 == 1:  # Movimientos horizontales (filas fijas, columnas cambian)
                found = False
                for col in range(cols):
                    if col != j and allocation[i][col] > 0:
                        if (i, col) not in visited:
                            j = col
                            path.append((i, j))
                            visited.add((i, j))
                            found = True
                            
                            break
                        

                            
                        
                if not found:
                    if path:
                        
                        visited.add((i, j))
                        path.pop()  # Retrocede a la celda anterior
                        i, j = path[-1]
                        

            else:  # Movimientos verticales (columnas fijas, filas cambian)
                found = False
                for row in range(rows):
                    if row != i and allocation[row][j] > 0:
                        if (row, j) not in visited:
                            i = row
                            path.append((i, j))
                            visited.add((i, j))
                            found = True
                            break

                        
                if not found:
                    
                    for row in range(rows):
                        if row != i and allocation[row][j] == 0:
                            if (row, j) not in visited:
                                i = row
                                path.append((row, j))
                                visited.add((row, j))
                                found = True
                                break

                            elif (row, j) == path[0]:
                                i = row
                                break
                            else:
                                path.pop()
                                i, j = path[-1]
                                break
                            

            # Si has vuelto a la celda de inicio, cierra el ciclo
            if path[0] == (i, j):
                break

        return path  # Asegurar que es un ciclo válido

    def update_allocation(self, path, allocation):
        
        min_value = min(allocation[i][j] for i, j in path[1::2])
        for k, (i, j) in enumerate(path):
            if k % 2 == 0:
                allocation[i][j] += min_value
            else:
                allocation[i][j] -= min_value
                
        return allocation

    def solve(self):
        # self.north_west_corner_method()
        self.least_cost_adjusted_method()
        
        

        while True:
            
            ui_stage_one, vj_stage_one, ui_stage_two, vj_stage_two = self.solve_fuzzy_dual()
            
            print("Variables fuzzy duales Stage One(ui, vj):", ui_stage_one, vj_stage_one)
            print("Variables fuzzy duales Stage Two(ui, vj):", ui_stage_two, vj_stage_two)
            
            deltas_stage_one = self.calculate_deltas(ui_stage_one, vj_stage_one, self.allocation_stage_one)
            deltas_stage_two = self.calculate_deltas(ui_stage_two, vj_stage_two, self.allocation_stage_two)
            print("Deltas para celdas no básicas Stage One:")
            pprint(deltas_stage_one)
            print("Deltas para celdas no básicas Stage Two:")
            pprint(deltas_stage_two)
            
            entering_cell_stage_one = self.find_most_negative_delta(deltas_stage_one)
            entering_cell_stage_two = self.find_most_negative_delta(deltas_stage_two)
            

            if entering_cell_stage_one is None and entering_cell_stage_two is None:
                print("Solución óptima encontrada.")

                rows_stage_one, cols_stage_one = self.allocation_stage_one.shape
                rows_stage_two, cols_stage_two = self.allocation_stage_two.shape
                matrix_result_stage_one = np.zeros((rows_stage_one, cols_stage_one), dtype=object)
                matrix_result_stage_two = np.zeros((rows_stage_two, cols_stage_two), dtype=object)

                pprint(self.allocation_stage_one)
                pprint(self.allocation_stage_two)

                for i in range(rows_stage_one):
                    for j in range(cols_stage_one):

                        matrix_result_stage_one[i][j] = self.tfn.constant_multiplication(self.allocation_stage_one[i][j], self.costs_stage_one[i][j])

                for i in range(rows_stage_two):
                    for j in range(cols_stage_two):

                        matrix_result_stage_two[i][j] = self.tfn.constant_multiplication(self.allocation_stage_two[i][j], self.costs_stage_two[i][j])


                optimus_solution_stage_one = ([0, 0, 0, 0], 1)
                optimus_solution_stage_two = ([0, 0, 0, 0], 1)

                for i in range(rows_stage_one):
                    for j in range(cols_stage_one):
                        optimus_solution_stage_one = self.tfn.sum_trapezoidal_fuzzy_numbers(optimus_solution_stage_one, matrix_result_stage_one[i][j])

                for i in range(rows_stage_two):
                    for j in range(cols_stage_two):
                        optimus_solution_stage_two = self.tfn.sum_trapezoidal_fuzzy_numbers(optimus_solution_stage_two, matrix_result_stage_two[i][j])

                optimus_solution = self.tfn.sum_trapezoidal_fuzzy_numbers(optimus_solution_stage_one, optimus_solution_stage_two)
                return optimus_solution, self.allocation_stage_one, self.allocation_stage_two

            elif entering_cell_stage_one is None or entering_cell_stage_two is None:
                if entering_cell_stage_two is None:
                    entering_cell = entering_cell_stage_one
                    allocation = self.allocation_stage_one

                    print(f"Seleccionando la celda {entering_cell} para entrarven:")
                    pprint(allocation)
                
                    path = self.find_closed_cycle(entering_cell, allocation)
                    print(f"Ciclo cerrado: {path}")
                    
                    self.allocation_stage_one = self.update_allocation(path, allocation)
                    print("Nueva asignación después de ajustar el ciclo:")
                    pprint(self.allocation_stage_one)

                else:
                    entering_cell = entering_cell_stage_two
                    allocation = self.allocation_stage_two

                    print(f"Seleccionando la celda {entering_cell} para entrar.")
                    pprint(allocation)
                
                    path = self.find_closed_cycle(entering_cell, allocation)
                    print(f"Ciclo cerrado: {path}")
                    
                    self.allocation_stage_two = self.update_allocation(path, allocation)
                    print("Nueva asignación después de ajustar el ciclo:")
                    pprint(self.allocation_stage_two)

            else:
                if deltas_stage_one[entering_cell_stage_one[0]][entering_cell_stage_one[1]] < deltas_stage_two[entering_cell_stage_two[0]][entering_cell_stage_two[1]]:
                    entering_cell = entering_cell_stage_one
                    allocation = self.allocation_stage_one

                    print(f"Seleccionando la celda {entering_cell} para entrarven:")
                    pprint(allocation)
                
                    path = self.find_closed_cycle(entering_cell, allocation)
                    print(f"Ciclo cerrado: {path}")
                    
                    self.allocation_stage_one = self.update_allocation(path, allocation)
                    print("Nueva asignación después de ajustar el ciclo:")
                    pprint(self.allocation_stage_one)

                else:
                    entering_cell = entering_cell_stage_two
                    allocation = self.allocation_stage_two

                    print(f"Seleccionando la celda {entering_cell} para entrar.")
                    pprint(allocation)
                
                    path = self.find_closed_cycle(entering_cell, allocation)
                    print(f"Ciclo cerrado: {path}")
                    
                    self.allocation_stage_two = self.update_allocation(path, allocation)
                    print("Nueva asignación después de ajustar el ciclo:")
                    pprint(self.allocation_stage_two)
                

class FuzzyCharts():
    
    
    def show_chart(self, list_tfn):
        i = 2 # producto 1
        j = 2 # origen 1
        self.list_tfn = list_tfn

        # Configurar matplotlib para usar LaTeX
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')  # Usar Computer Modern, que es la fuente por defecto de LaTeX
      
        x = np.linspace(min([solution[0][0] for solution in self.list_tfn]) - 5, max([solution[0][3] for solution in self.list_tfn]) + 5, 500)

        list_a = [solution[0][0] for solution in self.list_tfn]
        list_b = [solution[0][1] for solution in self.list_tfn]
        list_c = [solution[0][2] for solution in self.list_tfn]
        list_d = [solution[0][3] for solution in self.list_tfn]
        print(list_a)
        # Definir la función de pertenencia trapezoidal
        

        # Calcular los grados de pertenencia
        y = np.maximum(np.minimum(np.minimum((x-list_a[0])/(list_b[0]-list_a[0]), 1), (list_d[0]-x)/(list_d[0]-list_c[0])), 0)
        z = np.maximum(np.minimum(np.minimum((x-list_a[1])/(list_b[1]-list_a[1]), 1), (list_d[1]-x)/(list_d[1]-list_c[1])), 0)
        w = np.maximum(np.minimum(np.minimum((x-list_a[2])/(list_b[2]-list_a[2]), 1), (list_d[2]-x)/(list_d[2]-list_c[2])), 0)
        v = np.maximum(np.minimum(np.minimum((x-list_a[3])/(list_b[3]-list_a[3]), 1), (list_d[3]-x)/(list_d[3]-list_c[3])), 0)
        # Graficar
        plt.plot(x, y, label=f'Óptimo producto 1 ({list_a[0]}, {list_b[0]}, {list_c[0]}, {list_d[0]})')
        plt.plot(x, z, label=f'Óptimo producto 2 ({list_a[1]}, {list_b[1]}, {list_c[1]}, {list_d[1]})')
        plt.plot(x, w, label=f'Óptimo producto 3 ({list_a[2]}, {list_b[2]}, {list_c[2]}, {list_d[2]})')
        plt.plot(x, v, label=f'Óptimo global ({list_a[3]}, {list_b[3]}, {list_c[3]}, {list_d[3]})')

        plt.fill_between(x, y, alpha=0.2)
        plt.fill_between(x, z, alpha=0.2)
        plt.fill_between(x, w, alpha=0.2)
        plt.fill_between(x, v, alpha=0.2)
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


    # Ejemplo de uso
    # supply = [20, 30, 25]
    # demand = [10, 20, 30, 15]
    # costs = np.array([[8, 6, 10, 9], [9, 12, 13, 7], [14, 9, 16, 5]])

    ################################################################################################

    tfn = TrapezoidalFuzzyNumber()
    fuzzy_chart = FuzzyCharts()
    net = Network()
    # Vector de oferta para 3 productos (a_ip)
    # Paso 1: Definir los datos del problema
    supply = [
        [10, 14, 15],  # Oferta del primer proveedor para cada producto
        [5, 7, 8], # Oferta del segundo proveedor para cada producto
        [2, 3, 4]  # Oferta del tercer proveedor para cada producto
    ]

    # Vector de demanda para 3 productos (b_jp)
    demand = [
        [15, 14, 10], # Demanda del primer cliente para cada producto
        [8, 7, 5], # Demanda del segundo cliente para cada producto
        [4, 3, 2]   # Demanda del tercer cliente para cada producto
    ]

     
    # # Matriz de costos fuzzy para 3 productos (c_ijp)
    # costs_stage_one = [
    #     [  # Producto 1
    #         [([1, 4, 9, 19], 0.5), ([1, 2, 5, 9], 0.4), ([2, 5, 8, 18], 0.5)],
    #         [([8, 9, 12, 26], 0.5), ([3, 5, 8, 12], 0.2), ([7, 9, 13, 28], 0.4)],
    #         [([11, 12, 20, 27], 0.5), ([0, 5, 10, 15], 0.8), ([4, 5, 8, 11], 0.6)]
    #     ],
    #     [
    #         [([49, 50, 50, 50], 0.6), ([43, 46, 47, 48], 0.8), ([37, 57, 62, 64], 0.6)],
    #         [([71, 72, 73, 74], 0.8), ([107, 108, 108, 108], 0.6), ([60, 64, 65, 65], 0.6)],
    #         [([60, 75, 89, 98], 0.7), ([100, 101, 102, 102], 0.1), ([123, 128, 133, 136], 0.6)]
    #     ],
    #     [
    #         [([37, 38, 39, 43], 0.4), ([34, 50, 51, 51], 0.5), ([58, 59, 60, 60], 0.6)],
    #         [([87, 96, 99, 106], 0.4), ([114, 115, 115, 115], 0.5), ([59, 76, 77, 82], 0.5)],
    #         [([90, 91, 91, 91], 0.6), ([94, 99, 115, 122], 0.4), ([137, 139, 140, 140], 0.5)]],
    #     ]

    # costs_stage_two = [
    #     [  # Producto 1
    #         [([1, 4, 9, 19], 0.5), ([1, 2, 5, 9], 0.4), ([2, 5, 8, 18], 0.5)],
    #         [([8, 9, 12, 26], 0.5), ([3, 5, 8, 12], 0.2), ([7, 9, 13, 28], 0.4)],
    #         [([11, 12, 20, 27], 0.5), ([0, 5, 10, 15], 0.8), ([4, 5, 8, 11], 0.6)]
    #     ],
    #     [
    #         [([43, 49, 50, 50], 0.8), ([37, 39, 52, 54], 0.6), ([65, 67, 68, 69], 0.1)],
    #         [([56, 57, 57, 57], 0.7), ([23, 24, 25, 26], 0.3), ([27, 35, 40, 41], 0.4)],
    #         [([79, 87, 88, 95], 0.3), ([95, 104, 117, 118], 0.2), ([109, 113, 119, 120], 0.7)]
    #     ],
    #     [
    #         [([46, 47, 48, 48], 0.6), ([47, 50, 51, 54], 0.6), ([41, 60, 62, 64], 0.4)],
    #         [([61, 63, 73, 74], 0.5), ([32, 33, 34, 34], 0.4), ([53, 54, 57, 58], 0.5)],
    #         [([65, 68, 77, 79], 0.5), ([85, 87, 88, 89], 0.6), ([89, 90, 91, 91], 0.6)]
    #     ]
    # ]

    costs_stage_one = [
                        [  # Producto 1
                            [([1, 4, 9, 19], 0.5), ([1, 2, 5, 9], 0.4), ([2, 5, 8, 18], 0.5)],
                            [([8, 9, 12, 26], 0.5), ([3, 5, 8, 12], 0.2), ([7, 9, 13, 28], 0.4)],
                            [([11, 12, 20, 27], 0.5), ([0, 5, 10, 15], 0.8), ([4, 5, 8, 11], 0.6)]
                        ],
                        [
                            [([19, 21, 23, 25], 0.3), ([9, 12, 14, 16], 0.4), ([25, 27, 29, 31], 0.3)],
                            [([24, 26, 30, 33], 0.7), ([32, 35, 37, 41], 0.4), ([17, 19, 21, 23], 0.6)],
                            [([31, 35, 37, 39], 0.7), ([16, 18, 21, 23], 0.5), ([24, 28, 30, 32], 0.3)]
                        ],
                        [
                            [([30, 34, 38, 42], 0.8), ([47, 51, 58, 62], 0.8), ([68, 72, 76, 80], 0.7)],
                            [([72, 76, 80, 84], 0.8), ([67, 71, 78, 83], 0.2), ([89, 93, 97, 101], 0.6)],
                            [([53, 57, 61, 65], 0.3), ([87, 92, 96, 100], 0.4), ([106, 110, 115, 119], 0.7)]
                        ]
                        
                    ]
    
    costs_stage_two = [
                        [
                            [([9, 11, 13, 15], 0.2), ([10, 14, 16, 18], 0.3), ([17, 19, 21, 26], 0.7)],
                            [([2, 3, 4, 5], 0.4), ([4, 6, 7, 8], 0.8), ([6, 8, 10, 12], 0.4)],
                            [([8, 12, 14, 16], 0.6), ([17, 19, 21, 23], 0.6), ([5, 6, 7, 9], 0.5)]
                        ],
                        [
                            [([16, 18, 20, 22], 0.8), ([6, 8, 10, 13], 0.3), ([25, 27, 32, 34], 0.5)],
                            [([28, 30, 32, 34], 0.3), ([37, 39, 41, 43], 0.5), ([18, 20, 22, 24], 0.4)],
                            [([33, 35, 37, 39], 0.3), ([14, 17, 19, 21], 0.6), ([21, 23, 26, 29], 0.3)]
                        ],
                        [
                            [([21, 30, 33, 36], 0.5), ([34, 37, 40, 47], 0.4), ([60, 63, 66, 69], 0.6)],
                            [([53, 57, 60, 63], 0.6), ([53, 56, 60, 63], 0.6), ([67, 70, 73, 76], 0.4)],
                            [([30, 39, 44, 49], 0.4), ([66, 69, 72, 75], 0.8), ([77, 84, 87, 90], 0.3)]
                        ]
                    ]


    # costs_stage_one = [
    #                     [  # Ejemplo del libro
                        #     [([1, 4, 9, 19], 0.5), ([1, 2, 5, 9], 0.4), ([2, 5, 8, 18], 0.5)],
                        #     [([8, 9, 12, 26], 0.5), ([3, 5, 8, 12], 0.2), ([7, 9, 13, 28], 0.4)],
                        #     [([11, 12, 20, 27], 0.5), ([0, 5, 10, 15], 0.8), ([4, 5, 8, 11], 0.6)]
                        # ],
    #                     [
    #                         [tfn.generate_trapezoidal_fuzzy_number(10, 28), tfn.generate_trapezoidal_fuzzy_number(1, 19), tfn.generate_trapezoidal_fuzzy_number(19, 37)],
    #                         [tfn.generate_trapezoidal_fuzzy_number(19, 37), tfn.generate_trapezoidal_fuzzy_number(28, 46), tfn.generate_trapezoidal_fuzzy_number(10, 28)],
    #                         [tfn.generate_trapezoidal_fuzzy_number(28, 46), tfn.generate_trapezoidal_fuzzy_number(10, 28), tfn.generate_trapezoidal_fuzzy_number(19, 37)]
    #                     ],
    #                     [
                        #     [tfn.generate_trapezoidal_fuzzy_number(20, 56), tfn.generate_trapezoidal_fuzzy_number(38, 74), tfn.generate_trapezoidal_fuzzy_number(56, 92)],
                        #     [tfn.generate_trapezoidal_fuzzy_number(56, 92), tfn.generate_trapezoidal_fuzzy_number(56, 92), tfn.generate_trapezoidal_fuzzy_number(74, 110)],
                        #     [tfn.generate_trapezoidal_fuzzy_number(38, 74), tfn.generate_trapezoidal_fuzzy_number(74, 110), tfn.generate_trapezoidal_fuzzy_number(92, 128)]
                        # ]
    #                 ]
    
    # costs_stage_two = [
    # #                     [
    #                         [tfn.generate_trapezoidal_fuzzy_number(1, 19), tfn.generate_trapezoidal_fuzzy_number(8, 26), tfn.generate_trapezoidal_fuzzy_number(11, 27)],
    #                         [tfn.generate_trapezoidal_fuzzy_number(1, 9), tfn.generate_trapezoidal_fuzzy_number(3, 12), tfn.generate_trapezoidal_fuzzy_number(0, 15)],
    #                         [tfn.generate_trapezoidal_fuzzy_number(2, 18), tfn.generate_trapezoidal_fuzzy_number(7, 28), tfn.generate_trapezoidal_fuzzy_number(4, 11)]
    #                     ],
    #                     [
                        #     [tfn.generate_trapezoidal_fuzzy_number(10, 28), tfn.generate_trapezoidal_fuzzy_number(1, 19), tfn.generate_trapezoidal_fuzzy_number(19, 37)],
                        #     [tfn.generate_trapezoidal_fuzzy_number(19, 37), tfn.generate_trapezoidal_fuzzy_number(28, 46), tfn.generate_trapezoidal_fuzzy_number(10, 28)],
                        #     [tfn.generate_trapezoidal_fuzzy_number(28, 46), tfn.generate_trapezoidal_fuzzy_number(10, 28), tfn.generate_trapezoidal_fuzzy_number(19, 37)]
                        # ],
    #                     [
                        #     [tfn.generate_trapezoidal_fuzzy_number(15, 42), tfn.generate_trapezoidal_fuzzy_number(29, 56), tfn.generate_trapezoidal_fuzzy_number(42, 69)],
                        #     [tfn.generate_trapezoidal_fuzzy_number(42, 69), tfn.generate_trapezoidal_fuzzy_number(42, 69), tfn.generate_trapezoidal_fuzzy_number(56, 83)],
                        #     [tfn.generate_trapezoidal_fuzzy_number(29, 56), tfn.generate_trapezoidal_fuzzy_number(56, 83), tfn.generate_trapezoidal_fuzzy_number(69, 96)]
                        # ],
    #                 ]
  
    # # Vector de oferta para 3 productos (a_ip)
    # # Paso 1: Definir los datos del problema
    # supply = [
    #     [10, 10],  # Oferta del primer proveedor para cada producto
    #     [14, 14], # Oferta del segundo proveedor para cada producto
    #     [15, 15]  # Oferta del tercer proveedor para cada producto
    # ]

    # # Vector de demanda para 3 productos (b_jp)
    # demand = [
    # [15, 15], 
    # [14, 14],
    # [10, 10]
    # ]



    pprint(costs_stage_one)
    pprint(costs_stage_two)
    optimus_solution = ([0, 0, 0, 0], 1)
    list_solutions = []
    list_add_solutions = []
    list_allocation_stage_one = []
    list_allocation_stage_two = []
    for i in range(len(costs_stage_one)):

        # pprint(costs[i])
        #transportation_problem = KaurAndKumarMethod(traspose_supply[i], traspose_demand[i], costs_stage_one[i], costs_stage_two[i])
        transportation_problem = EbrahimnejadMethod(supply[i], demand[i], costs_stage_one[i], costs_stage_two[i])
        solution, allocation_stage_one, allocation_stage_two = transportation_problem.solve()
        list_solutions.append(solution)
        list_allocation_stage_one.append(allocation_stage_one)
        list_allocation_stage_two.append(allocation_stage_two)
        optimus_solution = tfn.sum_trapezoidal_fuzzy_numbers(optimus_solution, solution)
        list_add_solutions.append(optimus_solution)
        print('------------------------------------------------------------------------------------')
    list_solutions = list_solutions + [optimus_solution]
    pprint(optimus_solution)
    pprint(list_solutions)

    fuzzy_chart.show_chart(list_solutions)
    # net.show_network(list_allocation_stage_one[0], list_allocation_stage_two[0])



        