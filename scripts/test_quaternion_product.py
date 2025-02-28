import sympy as sp

class Quaternion:
    def __init__(self, w: sp.Symbol, x: sp.Symbol, y: sp.Symbol, z: sp.Symbol):
        self.w, self.x, self.y, self.z = w, x, y, z

    def to_matrix(self) -> sp.Matrix:
        return sp.Matrix([self.w, self.x, self.y, self.z])

def quaternion_product(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """
    Computes the Hamilton product of two quaternions q1 and q2.
    q1 and q2 are instances of the Quaternion class.
    """
    w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
    w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2 # type: ignore
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2 # type: ignore
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2 # type: ignore
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2 # type: ignore
    
    return Quaternion(w, x, y, z)

# Define symbolic quaternion components
q1 = Quaternion(*sp.symbols('w1 x1 y1 z1'))
q2 = Quaternion(*sp.symbols('w2 x2 y2 z2'))

# Compute quaternion product
q_prod = quaternion_product(q1, q2).to_matrix()

# Compute Jacobian with respect to the second quaternion
q2_vars = sp.Matrix([q2.w, q2.x, q2.y, q2.z])
jacobian_q2 = q_prod.jacobian(q2_vars)
jacobian_q2_transpose = jacobian_q2.T
jacobian_q2_transpose_inv = sp.simplify(jacobian_q2_transpose.inv())
jacobian_q2_transpose_inv = jacobian_q2_transpose_inv.subs(q1.w**2 + q1.x**2 + q1.y**2 + q1.z**2, 1) #type: ignore

# Substitute x_1^2 + x_2^2 + x_3^2 + x_4^2 = 1 wherever it appears

# Display results
print("Quaternion product:")
sp.pprint(q_prod)
print("\n[Q_^T] Jacobian with respect to the second quaternion:")
sp.pprint(jacobian_q2)
print("\n[Q_] Transpose of the Jacobian with respect to the second quaternion:")
sp.pprint(jacobian_q2_transpose)
print("\n[Q_^-1] Inverse of the transpose of the Jacobian with respect to the second quaternion:")
sp.pprint(jacobian_q2_transpose_inv)



# Compute the angular velocity as:
# Omega = [0, w_x, w_y, w_z].T = 2 * q @ q_dot, where @ is the Hamilton product = 2 * Q_^T * q_dot
q = Quaternion(*sp.symbols('w x y z'))
q_dot = Quaternion(*sp.symbols('w_dot x_dot y_dot z_dot'))

q_times_q_dot = quaternion_product(q, q_dot).to_matrix()
Q_T = q_times_q_dot.jacobian(q_dot.to_matrix())

omega = 2 * q_times_q_dot
omega_check = 2 * Q_T * q_dot.to_matrix()

print("\nAngular velocity:")
sp.pprint(omega)

assert omega == omega_check, "The angular velocity is computed incorrectly."


# Differentiate omega wrt time, considering q(t) and q_dot(t), to get the angular acceleration
# Omega_dot = [0, w_x_dot, w_y_dot, w_z_dot].T = 2 * q_dot @ q_dot + 2 * q @ q_dot_dot
q_dot_dot = Quaternion(*sp.symbols('w_dot_dot x_dot_dot y_dot_dot z_dot_dot'))

q_dot_times_q_dot = quaternion_product(q_dot, q_dot).to_matrix()
q_times_q_dot_dot = quaternion_product(q, q_dot_dot).to_matrix()
Q_T_dot = q_dot_times_q_dot.jacobian(q_dot.to_matrix())
Q_T_plus = q_times_q_dot_dot.jacobian(q_dot_dot.to_matrix())

omega_dot = 2 * q_dot_times_q_dot + 2 * q_times_q_dot_dot
omega_dot_check = 2 * Q_T_plus * q_dot_dot.to_matrix() + 2 * Q_T_dot * q_dot.to_matrix()
omega_dot_legnani = sp.simplify(2 * Q_T * (q_dot_dot.to_matrix() - Q_T_dot.T * Q_T * q_dot.to_matrix()))

print("\nAngular acceleration:")
sp.pprint(omega_dot)
print("\nAngular acceleration (check):")
sp.pprint(omega_dot_check)
print("\nAngular acceleration (Legnani):")
sp.pprint(omega_dot_legnani)

assert omega_dot == omega_dot_check, "The angular acceleration is computed incorrectly."
assert omega_dot == omega_dot_legnani, "The angular acceleration is different from the one computed by Legnani."

