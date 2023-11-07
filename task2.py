import numpy as np
from numpy.linalg import norm


def main():
    # Read the objective function
    variables_count = int(input("Enter the number of variables: "))
    C = [int(x) for x in input("Enter the coefficients of the objective function: ").split()]
    is_min = input("Maximize or minimize? (max/min): ") == "min"
    if is_min:
        C = [-c for c in C]

    # Read the constraints
    constraints_count = int(input("Enter the number of constraints: "))
    slack_count = constraints_count
    A = []
    for i in range(constraints_count):
        A.append([int(x) for x in input(f"Enter the coefficients of the constraint {i + 1}: ").split()])
    b = [int(x) for x in input("Enter the right hand sides of the constraints: ").split()]
    eps = float(input("Enter the precision (e.g. 0.001): "))

    # Enter initial point
    x = [float(x) for x in input("Enter the initial point: ").split()]

    # Set the precision for printing
    precision = int(np.log10(1 / eps))
    np.set_printoptions(precision=precision, suppress=True)

    # Show the problem
    if is_min:
        print("\nMinimize:")
    else:
        print("\nMaximize:")
    objective = [f"{c * (-1 if is_min else 1)}*x{i + 1}" for i, c in enumerate(C)]
    objective = " + ".join(objective)
    print(f"z = {objective}")
    print("Subject to:")
    for i in range(constraints_count):
        constraint = [f"{c}*x{i + 1}" for i, c in enumerate(A[i])]
        constraint = " + ".join(constraint)
        print(f"{constraint} <= {b[i]}")
    print("x >= 0")
    print(f"Initial point: x = {x}")

    # Check if the method is applicable
    if not np.all(np.array(b) >= 0):
        print("\nThe method is not applicable because not all b >= 0.")
        return

    # Solve the problem using different methods
    print("\nSIMPLEX METHOD:\n")
    x1, z1 = simplex(constraints_count, variables_count, slack_count, C, A, b, eps, precision)
    print("\nINTERIOR POINT METHOD with alpha = 0.5:\n")
    x2, z2 = interior_point(x, 0.5, A, C, constraints_count, variables_count, slack_count, eps, precision)
    print("\nINTERIOR POINT METHOD with alpha = 0.9:\n")
    x3, z3 = interior_point(x, 0.9, A, C, constraints_count, variables_count, slack_count, eps, precision)

    print("\nRESULTS:\n")
    print(f"Simplex method solution: x = {x1}, z = {z1}")
    print(f"Interior point method solution with alpha = 0.5: x = {x2[:variables_count]}, z = {z2}")
    print(f"Interior point method solution with alpha = 0.9: x = {x3[:variables_count]}, z = {z3}")


def interior_point(x, alpha, A, c, constraints_count, variables_count, slack_count, eps, precision):
    # make the matrix c with the slack variables
    c_matrix = np.zeros(variables_count + slack_count)
    c_matrix[:variables_count] = c
    c = c_matrix.transpose()

    # make the matrix A with the slack variables
    matrix = np.zeros((constraints_count, variables_count + slack_count))
    matrix[:, :variables_count] = A
    matrix[:, variables_count:] = np.identity(slack_count)
    A = matrix

    i = 1
    while True:
        # Make iteration of the interior point method
        v = x
        D = np.diag(x)
        AA = A @ D
        cc = D @ c
        I = np.eye(len(c))
        F = AA @ np.transpose(AA)
        FI = np.linalg.inv(F)
        H = np.transpose(AA) @ FI
        P = np.subtract(I, np.dot(H, AA))
        cp = P @ cc
        nu = np.absolute(np.min(cp))
        y = np.ones(len(c), float) + (alpha / nu) * cp
        yy = D @ y
        x = yy

        print("In iteration ", i, " we have x = ", x)
        i = i + 1

        if norm(np.subtract(yy, v), ord=2) < eps:
            break

    # Compute objective value
    objective_value = np.round(c @ x, precision)
    if -objective_value == objective_value:
        objective_value = 0.0

    # Print the results
    print("In the last iteration ", i, " we have x = ", x)
    print("z = ", objective_value)

    return x, objective_value


def simplex(constraints_count: int, variables_count: int, slack_count: int, C: list, A: list, b: list, eps: float,
            precision: int):
    # Construct the initial tableau
    matrix = np.zeros((constraints_count + 1, variables_count + slack_count + 1))
    matrix[0, :variables_count] = C
    matrix[1:, :variables_count] = A
    matrix[1:, variables_count:-1] = np.identity(slack_count)
    matrix[1:, -1] = np.array(b)
    print(f"Initial matrix:\n{matrix}")

    # Iterate until the solution is optimal
    i = 0
    while not np.all(matrix[0, :-1] <= eps):
        i += 1
        print(f"\nIteration {i}.")

        pivot_column = np.argmax(matrix[0, :-1])
        print(f"Pivot: col {pivot_column}; ", end="")

        ratios = []
        for row in matrix[1:]:
            if row[pivot_column] > eps:
                ratios.append(row[-1] / row[pivot_column])
            else:
                ratios.append(np.inf)
        pivot_row = np.argmin(ratios) + 1
        print(f"row {pivot_row}; ", end="")

        pivot = matrix[pivot_row, pivot_column]
        print(f"val {np.round(pivot, precision)}")

        matrix[pivot_row] /= pivot
        for row in range(matrix.shape[0]):
            if row != pivot_row:
                matrix[row] -= matrix[row, pivot_column] * matrix[pivot_row]
        print(f"New matrix:\n{matrix}")

        if i > variables_count + constraints_count:
            break

    # Find decision variables from the matrix
    decision_variables = [0 for x in range(variables_count)]
    for column in range(variables_count):
        for row in range(1, constraints_count + 1):
            if matrix[row, column] == 1:
                decision_variables[column] = np.round(matrix[row, -1], precision)
            elif matrix[row, column] != 0:
                decision_variables[column] = 0
                break

    # Find the value of the objective function
    objective_value = np.round(-matrix[0, -1], precision)
    if -objective_value == objective_value:
        objective_value = 0.0

    # Print the results
    print(f"\nx = {decision_variables}")
    print(f"z = {objective_value}")

    return decision_variables, objective_value


if __name__ == "__main__":
    main()
