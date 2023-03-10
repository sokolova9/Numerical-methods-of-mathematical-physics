import numpy as np
import sympy as sym

x = sym.Symbol('x')

def int_i(h, a_1, b_1, f):
    phi = (sym.integrate(f, (x, a_1, b_1)))/h
    return phi

def get_lhs(alpha1, alpha2, d, d0, dn,a,n,h):
    lhs = np.zeros((n, n))
    for row in range(1, n-1):
        lhs[row][row-1] = -a[row-1]
        lhs[row][row] = a[row]+a[row-1]+d[row-1]*h**2
        lhs[row][row+1] = -a[row]
    lhs[0][0] = a[0]+h*(alpha1+h/2*d0)
    lhs[0][1] = -a[0]
    lhs[n - 1][-2] = -a[-1]
    lhs[n - 1][-1] = a[-1]+h*(alpha2+h/2*dn)
    return lhs


def get_rhs(m_1, m_2,phi, phi_0, phi_n, h, n):
    rhs = np.zeros(n)
    rhs[0] = h*(m_1+h/2*phi_0)
    for i in range(0, n-2):
        rhs[i+1]=phi[i]*h**2
    rhs[n-1] = h*(m_2+h/2*phi_n)
    return rhs


def solve(m1, m2, alpha1, alpha2, n, a, b ):
    k = sym.parse_expr("1")
    q = sym.parse_expr("1")
    f = sym.parse_expr("3*E**(-x)-2")
    h = (b - a) / n
    A = lambda u: -sym.diff(u, x, 2) + q*u
    x_i = [i * h for i in range(n + 1)]

    a_i = [1/int_i(h, i*h, (i+1)*h, 1/k) for i in range(1, n+1)]
    d_i = [int_i(h, i*h-h/2, i*h+h/2, q) for i in range(1, n)]
    phi_i = [int_i(h, i*h-h/2, i*h+h/2, f) for i in range(1, n)]

    d_0 = 2*int_i(h, 0, h/2, q)
    phi_0 = 2*int_i(h, 0, h/2, f)

    d_n = 2*int_i(h, b-h/2, b, q)
    phi_n = 2 * int_i(h, b-h/2, b, f)

    lhs = get_lhs(alpha1, alpha2,  d_i, d_0, d_n,a_i,n+1,  h)
    rhs = get_rhs(m1, m2,phi_i, phi_0, phi_n, h, n+1)
    y = np.linalg.solve(lhs, rhs)

    p = 7
    u_i = np.polyfit(x_i, y, p)
    u = sum(u_i[i]*x**(p-i) for i in range(p+1))

    print("u = ", u)

    plot = sym.plot(A(u)-f, (x, 0, 1), show=False)
    plot.show()

    return u

def main():
    alpha1 = 0
    alpha2 = 1
    m1 = 0
    m2 = 0
    n = 5
    a = 0
    b = 1
    solve(m1, m2, alpha1, alpha2, n, a, b )

main()
