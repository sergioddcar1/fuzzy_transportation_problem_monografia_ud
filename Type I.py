import numpy as np
from pprint import pprint



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




    def __init__(self, supply, demand, costs):
        self.supply = supply.copy()
        self.demand = demand.copy()
        self.costs = costs
        self.allocation = np.zeros((len(supply), len(demand)))
        self.tfn = TrapezoidalFuzzyNumber()

    def adjust_costs(self):

        rows, cols = self.allocation.shape
        adjusted_costs = np.zeros((rows, cols))
        min_membership = 1

        for i in range(rows):
            for j in range(cols):
                if self.costs[i][j][1] < min_membership:
                    min_membership = self.costs[i][j][1]
                    
        for i in range(rows):
            for j in range(cols):
                adjusted_costs[i][j] = min_membership*(sum(self.costs[i][j][0])/4)
        
        return adjusted_costs

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

        print("Asignación inicial (método de la Esquina Noroccidental):")
        pprint(self.allocation)

    def least_cost_method(self):
        
        adjusted_costs = self.adjust_costs()
        
        # Bucle mientras haya oferta o demanda pendiente
        while np.any(self.supply) and np.any(self.demand):
            
            # Encontrar el índice de la celda con el costo más bajo
            min_cost_index = np.unravel_index(np.argmin(adjusted_costs, axis=None), adjusted_costs.shape)
            i, j = min_cost_index
            
            # Determinar la cantidad a asignar (mínimo entre oferta y demanda disponible)
            allocation_min = min(self.supply[i], self.demand[j])
            self.allocation[i][j] = allocation_min
            
            # Actualizar la oferta y la demanda
            self.supply[i] -= allocation_min
            self.demand[j] -= allocation_min
            
            # Si la oferta se ha agotado, eliminar la fila correspondiente
            if self.supply[i] == 0:
                adjusted_costs[i, :] = np.inf
                
            
            # Si la demanda se ha agotado, eliminar la columna correspondiente
            if self.demand[j] == 0:
                adjusted_costs[:, j] = np.inf
                
        print("Asignación inicial (método de Costo Mínimo):")
        pprint(self.allocation)

    def solve_fuzzy_dual(self):
        rows, cols = self.allocation.shape
        ui = np.full(rows, None)  # Variables fuzzy para las filas
        vj = np.full(cols, None)  # Variables fuzzy para las columnas
        adjusted_costs = self.adjust_costs()
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
                    ui[i] = 0  # Asignamos las variables unicas basicas como 0
            
            
        else:
            ui[0] = 0
        
        while None in ui or None in vj:
            for i in range(rows):
                for j in range(cols):
                    if self.allocation[i][j] > 0:  # Solo celdas básicas
                        if ui[i] is not None and vj[j] is None:
                            vj[j] = adjusted_costs[i][j] - ui[i]
                        elif vj[j] is not None and ui[i] is None:
                            ui[i] = adjusted_costs[i][j] - vj[j]

        return ui, vj

    def calculate_deltas(self, ui, vj):
        rows, cols = self.allocation.shape
        deltas = np.zeros((rows, cols))
        adjusted_costs = self.adjust_costs()

        for i in range(rows):
            for j in range(cols):
                if self.allocation[i][j] == 0:  # Solo para celdas no básicas
                    deltas[i][j] = adjusted_costs[i][j] - (ui[i] + vj[j])

        return deltas

    def find_most_negative_delta(self, deltas):
        min_value = np.min(deltas)
        if min_value >= 0:
            return None  # La solución es óptima
        return np.unravel_index(np.argmin(deltas), deltas.shape)  # Retorna la celda con el delta más negativo

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
        # self.north_west_corner_method()
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
                pprint(self.allocation)
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




# Ejemplo de uso
# supply = [20, 30, 25]
# demand = [10, 20, 30, 15]
# costs = np.array([[8, 6, 10, 9], [9, 12, 13, 7], [14, 9, 16, 5]])

################################################################################################

# Caso del libro


# Vector de oferta (a_i)
supply = [10, 14, 15]

# Vector de demanda (b_j)
demand = [15, 14, 10]

# # Matriz de costos fuzzy representada como una tupla (lista de 4 elementos, número)
costs = [  
        [([1, 4, 9, 19], 0.5), ([1, 2, 5, 9], 0.4), ([2, 5, 8, 18], 0.5)],
        [([8, 9, 12, 26], 0.5), ([3, 5, 8, 12], 0.2), ([7, 9, 13, 28], 0.4)],
        [([11, 12, 20, 27], 0.5), ([0, 5, 10, 15], 0.8), ([4, 5, 8, 11], 0.6)]
    ]

########################################################################
# Caso especial 
# # Vector de oferta (a_i)
# supply = [40, 35, 30]

# # Vector de demanda (b_j)
# demand = [50, 25, 30]

# # Matriz de costos fuzzy representada como una tupla (lista de 4 elementos, número)
# costs = [
#     [([2, 3, 6, 11], 0.3), ([3, 6, 8, 14], 0.2), ([1, 3, 5, 7], 0.6)],
#         [([5, 6, 9, 18], 0.7), ([2, 4, 7, 10], 0.4), ([6, 8, 11, 15], 0.5)],
#         [([9, 10, 14, 22], 0.6), ([3, 4, 6, 8], 0.3), ([7, 9, 12, 19], 0.8)]
# ]

# Caso especial 2
# # Vector de oferta (a_i)
# supply = [12, 10, 15]

# # Vector de demanda (b_j)
# demand = [17, 12, 8]

# # Matriz de costos fuzzy representada como una tupla (lista de 4 elementos, número)
# costs = [  # Producto 2
#             [([2, 3, 6, 11], 0.3), ([3, 6, 8, 14], 0.2), ([1, 3, 5, 7], 0.6)],
#             [([5, 6, 9, 18], 0.7), ([2, 4, 7, 10], 0.4), ([6, 8, 11, 15], 0.5)],
#             [([9, 10, 14, 22], 0.6), ([3, 4, 6, 8], 0.3), ([7, 9, 12, 19], 0.8)]
#         ]

# Caso especial 3
# # Vector de oferta (a_i)
# supply = [8, 11, 19]

# # Vector de demanda (b_j)
# demand = [10, 13, 15]

# # Matriz de costos fuzzy representada como una tupla (lista de 4 elementos, número)
# costs = [  # Producto 3
#         [([4, 6, 10, 17], 0.4), ([2, 4, 6, 10], 0.3), ([5, 7, 10, 15], 0.5)],
#         [([7, 8, 11, 20], 0.6), ([4, 5, 8, 12], 0.5), ([8, 10, 13, 18], 0.7)],
#         [([3, 5, 9, 13], 0.8), ([1, 3, 6, 9], 0.4), ([6, 7, 9, 14], 0.6)]
#         ]



transportation_problem = KaurAndKumarMethod(supply, demand, costs)
#transportation_problem = EbrahimnejadMethod(supply, demand, costs)
pprint(transportation_problem.solve())
