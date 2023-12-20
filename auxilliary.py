from numba import jit
import numpy as np

@jit(nopython=True)
def rk4_c(q, p, qf, w, dt):
    fq, fp = qf
    k1 = dt * (p + fp)
    l1 = -dt * (w ** 2 * q + fq)  # [wn2] is w_alpha ^ 2
    k2 = dt * ((p + 0.5 * l1) + fp)
    l2 = -dt * (w ** 2 * (q + 0.5 * k1) + fq)
    k3 = dt * ((p + 0.5 * l2) + fp)
    l3 = -dt * (w ** 2 * (q + 0.5 * k2) + fq)
    k4 = dt * ((p + l3) + fp)
    l4 = -dt * (w ** 2 * (q + k3) + fq)
    q = q + 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    p = p + 0.166667 * (l1 + 2 * l2 + 2 * l3 + l4)
    return q, p

@jit(nopython=True)
def rk4_c(h, psi, dt):
    k1 = (-1j * h.dot(psi))
    k2 = (-1j * h.dot(psi + 0.5 * dt * k1))
    k3 = (-1j * h.dot(psi + 0.5 * dt * k2))
    k4 = (-1j * h.dot(psi + dt * k3))
    psi = psi + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi