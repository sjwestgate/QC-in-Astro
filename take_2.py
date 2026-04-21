# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:54:48 2026

@author: swest
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DilutionRefrigerator:
    def __init__(self, n_dot_He3):
        self.n_dot = n_dot_He3

    def cooling_power(self, T_mc, T_in):
        return self.n_dot * (95 * T_mc**2 - 11 * T_in**2)

    def heat_load(self, T_mc):
        return 1e-6 + 5e-6 * T_mc

    def steady_state_temperature(self, T_in, T_guess=0.1):
        T = T_guess

        for _ in range(1000):
            Q_cool = self.cooling_power(T, T_in)
            Q_load = self.heat_load(T)

            T = T - 0.1 * (Q_cool - Q_load)
            T = max(T, 0.001)

        return T


class CounterflowHEX:
    def __init__(self, U, A):
        self.U = U
        self.A = A

    def exchange(self, T_hot_in, T_cold_in, C_hot, C_cold):
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        Cr = C_min / C_max

        NTU = self.U * self.A / C_min

        if Cr != 1:
            eff = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
        else:
            eff = NTU / (1 + NTU)

        Q_max = C_min * (T_hot_in - T_cold_in)
        Q = eff * Q_max

        T_hot_out = T_hot_in - Q / C_hot
        T_cold_out = T_cold_in + Q / C_cold

        return T_hot_out, T_cold_out


def evaluate_design(n_dot, UA, heat_leak_base):
    fridge = DilutionRefrigerator(n_dot_He3=n_dot)

    def custom_heat_load(T):
        return heat_leak_base + 5e-6 * T

    fridge.heat_load = custom_heat_load

    hx = CounterflowHEX(U=UA, A=1.0)

    T_still = 0.7  # K
    T_mc = 0.1     # initial guess

    for _ in range(50):
        C_hot = 1e-3
        C_cold = 1e-3

        T_cold_in = T_mc

        T_hot_out, _ = hx.exchange(T_still, T_cold_in, C_hot, C_cold)
        T_in = T_hot_out

        T_mc_new = fridge.steady_state_temperature(T_in)

        T_mc = 0.5 * T_mc + 0.5 * T_mc_new

    Q_100mK = fridge.cooling_power(0.1, T_in)

    return T_mc, T_in, Q_100mK


if __name__ == "__main__":
    flows = np.linspace(1e-5, 5e-4, 20)
    UAs = np.logspace(0.01, 1.0, 50)

    results = []

    for n in flows:
        for UA in UAs:
            T_mc, T_in, Q = evaluate_design(n, UA, heat_leak_base=1e-6)

            results.append({
                "flow": n,
                "UA": UA,
                "T_mc": T_mc,
                "T_in": T_in,
                "Q_100mK": Q
            })

    df = pd.DataFrame(results)
    fixed_quantity = "flow"
    target = 3e-5
    closest = df.iloc[(df[fixed_quantity] - target).abs().idxmin()][fixed_quantity]
    subset = df[df[fixed_quantity] == closest]
    
    y = "T_mc"
    x = "T_in"

    plt.scatter(subset[x], subset[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()