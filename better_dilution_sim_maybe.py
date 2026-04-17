# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:07:24 2026

@author: swest
"""

import numpy as np
import matplotlib.pyplot as plt

class DilutionRefrigerator:
    def __init__(self, n_dot_He3):
        """
        n_dot_He3: molar flow rate of He-3 (mol/s)
        """
        self.n_dot = n_dot_He3

    def cooling_power(self, T_mc, T_in):
        """
        Cooling power at mixing chamber (Watts)
        Approximation valid below ~0.3 K
        """
        return self.n_dot * (84 * T_mc**2 - 20 * T_in**2)

    def heat_load(self, T_mc):
        """
        External heat load (W)
        You can customize this depending on your system
        """
        return 1e-6 + 5e-6 * T_mc

    def steady_state_temperature(self, T_in, T_guess=0.1):
        """
        Solve for steady-state mixing chamber temperature
        where cooling power = heat load
        """
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


def evaluate_design(n_dot, T_in, UA, heat_leak_base):
    fridge = DilutionRefrigerator(n_dot_He3=n_dot)

    def custom_heat_load(T):
        return heat_leak_base + 5e-6 * T

    fridge.heat_load = custom_heat_load

    T_mc = fridge.steady_state_temperature(T_in)
    Q_100mK = fridge.cooling_power(0.1, T_in)

    return T_mc, Q_100mK


if __name__ == "__main__":
    # Original example
    fridge = DilutionRefrigerator(n_dot_He3=1e-4)
    T_incoming = 0.05
    T_mc = fridge.steady_state_temperature(T_incoming)

    print(f"Mixing chamber temperature: {T_mc:.4f} K")

    flows = np.linspace(1e-5, 5e-4, 50)
    temps = []

    for f in flows:
        fridge = DilutionRefrigerator(n_dot_He3=f)
        temps.append(fridge.steady_state_temperature(T_incoming))

    plt.plot(flows, temps)
    plt.xlabel("He-3 flow rate (mol/s)")
    plt.ylabel("Mixing chamber temperature (K)")
    plt.title("Dilution Refrigerator Performance")
    plt.show()

    # Optimisation sweep
    flows = np.linspace(1e-5, 5e-4, 30)
    Tins = np.linspace(0.02, 0.1, 20)
    UAs = np.linspace(0.01, 1.0, 20)

    results = []

    for n in flows:
        for Tin in Tins:
            for UA in UAs:
                T_mc, Q = evaluate_design(n, Tin, UA, heat_leak_base=1e-6)

                results.append({
                    "flow": n,
                    "Tin": Tin,
                    "UA": UA,
                    "T_mc": T_mc,
                    "Q_100mK": Q
                })

import pandas as pd

df = pd.DataFrame(results)

target = 0.05
closest_Tin = df.iloc[(df["Tin"] - target).abs().idxmin()]["Tin"]
subset = df[df["Tin"] == closest_Tin]

plt.scatter(subset["Q_100mK"], subset["T_mc"])
plt.xlabel("Q at 100 mK")
plt.ylabel("Mixing chamber temperature")
plt.show()