import numpy as np


def braket(state1, operator, state2):
    """Calculate the matrix element <state1|operator|state2>."""
    return np.vdot(state1.vector, np.dot(operator.matrix, state2.vector))


class SpinOperator:
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = matrix

    def __repr__(self):
        return f"SpinOperator(name={self.name})"

    def __str__(self):
        return f"SpinOperator: {self.name}\nMatrix:\n{self.matrix}"

    def __mul__(self, other):
        if isinstance(other, SpinOperator):
            new_name = f"({self.name} * {other.name})"
            new_matrix = np.dot(self.matrix, other.matrix)
            return SpinOperator(new_name, new_matrix)
        elif isinstance(other, SpinState):
            new_name = f"{self.name}{other.name}"
            new_vector = np.dot(self.matrix, other.vector)
            return SpinState(new_name, new_vector)
        elif isinstance(other, (int, float, complex)):
            return SpinOperator(f"({other}*{self.name})", other * self.matrix)
        else:
            return NotImplemented

    def __rmul__(self, other):
        # Handles scalar * Operator
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        elif isinstance(other, SpinState):
            new_state = np.vdot(other.vector, self.matrix)
            return SpinState(f"{other.name}{self.name}", new_state)

    def __add__(self, other):
        if isinstance(other, SpinOperator):
            new_name = f"({self.name} + {other.name})"
            new_matrix = self.matrix + other.matrix
            return SpinOperator(new_name, new_matrix)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, SpinOperator):
            new_name = f"({self.name} - {other.name})"
            new_matrix = self.matrix - other.matrix
            return SpinOperator(new_name, new_matrix)
        elif isinstance(other, (int, float, complex)):
            return SpinOperator(f"({self.name} - {other})", self.matrix - other * np.eye(self.matrix.shape[0]))
        else:
            return NotImplemented

    def __neg__(self):
        new_name = f"-{self.name}"
        new_matrix = -self.matrix
        return SpinOperator(new_name, new_matrix)

    def __pow__(self, power):
        if isinstance(power, int) and power >= 0:
            new_name = f"({self.name}**{power})"
            new_matrix = np.linalg.matrix_power(self.matrix, power)
            return SpinOperator(new_name, new_matrix)
        else:
            raise ValueError("Power must be a non-negative integer")


class State:
    def __init__(self, name, vector):
        self.name = name
        self.vector = vector

    def __repr__(self):
        return f"State(name={self.name})"

    def __str__(self):
        return f"State: {self.name}\nVector:\n{self.vector}"


class SpinState(State):
    pass


class PhononState:
    pass


S_plus = SpinOperator("S+", np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))

S_minus = SpinOperator("S-", np.sqrt(2) * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

S_z = SpinOperator("S_z", np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]))

S_x = (S_plus + S_minus) * 0.5
S_y = (S_plus - S_minus) * (0.5j)

F_x = 0.5 * (S_plus**2 + S_minus**2)
F_y = -0.5j * (S_plus**2 - S_minus**2)
F_xp = np.sqrt(2) * (S_x * S_z + S_z * S_x)
F_yp = np.sqrt(2) * (S_y * S_z + S_z * S_y)
F_z = 3 * (S_z**2 - 2 / 3)

if __name__ == "__main__":
    print("Available operators:")
    print(f"F_x: {F_x}")
    print(f"F_y: {F_y}")
    print(f"F_z: {F_z}")
    print(f"F_xp: {F_xp}")
    print(f"F_yp: {F_yp}")

    s_p1 = SpinState("|m_s=1>", np.array([1, 0, 0]))
    s_0 = SpinState("|m_s=0>", np.array([0, 1, 0]))
    s_m1 = SpinState("|m_s=-1>", np.array([0, 0, 1]))

    braket_example = braket(s_0, F_z, s_0)
    print(f"Braket example: {braket_example}")
