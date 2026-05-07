# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:56:27 2026

@author: swest
"""


import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct
from pathlib import Path

OUTPUT_DIR = Path(".")

# ── Physical constants ────────────────────────────────────────────────
R_GAS     = 8.314           # J/molK
A_DIL     = 95.0            # 
A_CON     = 11.0            # I keep forgetting if its 95 11 or 96 12
r_K_COEFF = 0.02            # 
A_MC      = 10.0           # Very big surface area: sintered silver
Q_EXT     = 1e-6            #   parasitic heat load

#tasty lipids mmmmmmmm   <-   David wrote this, his project's on lipids. idk

def hx_effectiveness(T_inlet, n3_dot, UA):
    if UA == 0:
        return 0.0
    C_hot = n3_dot * 2.0 * A_CON * R_GAS * T_inlet ## Heat capacitance - comes from differentiating enthalpy
    NTU   = UA / C_hot
    return 1.0 - np.exp(-NTU)  ## epsilon


def solve_Tmc(T_inlet, n3_dot, UA):
    eps     = hx_effectiveness(T_inlet, n3_dot, UA)
    alpha_h = 1.0 - eps
    beta_h  = eps * T_inlet

    A_coef =  n3_dot * R_GAS * (A_DIL - A_CON * alpha_h**2)
    B_coef = -n3_dot * R_GAS *  2.0 * A_CON * alpha_h * beta_h
    C_coef = -n3_dot * R_GAS *  A_CON * beta_h**2 - Q_EXT

    if A_coef <= 0:
        return np.nan, np.nan, np.nan

    discriminant = B_coef**2 - 4.0 * A_coef * C_coef
    if discriminant < 0:
        return np.nan, np.nan, np.nan

    sqrt_disc = np.sqrt(discriminant)
    T1 = (-B_coef - sqrt_disc) / (2.0 * A_coef)
    T2 = (-B_coef + sqrt_disc) / (2.0 * A_coef)

    # Pick smallest positive root
    candidates = [t for t in (T1, T2) if t > 0]
    if not candidates:
        return np.nan, np.nan, np.nan
    T_mc = min(candidates)

    if T_mc >= T_inlet:
        return np.nan, np.nan, np.nan

    T_hx   = alpha_h * T_mc + beta_h
    Q_cool = n3_dot * R_GAS * (A_DIL * T_mc**2 - A_CON * T_hx**2)

    # Kapitza correction
    R_K     = r_K_COEFF / (A_MC * T_mc**3)
    T_mc_eff = T_mc + Q_cool * R_K

    return T_mc, Q_cool, T_mc_eff



T_inlets        = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
UA_values       = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
n3_dots         = np.logspace(-5, -3, 200)   # 10 μmol/s → 1 mmol/s

UA_fixed = 0.01
Ti_fixed = 1.0


results_Tinlet = {}
for Ti in T_inlets:
    Tmc_arr, Q_arr, Teff_arr = [], [], []
    for nd in n3_dots:
        Tmc, Q, Teff = solve_Tmc(Ti, nd, UA_fixed)
        Tmc_arr.append(Tmc); Q_arr.append(Q); Teff_arr.append(Teff)
    results_Tinlet[Ti] = (np.array(Tmc_arr), np.array(Q_arr), np.array(Teff_arr))

results_UA = {}
for UA in UA_values:
    Tmc_arr, Q_arr, Teff_arr = [], [], []
    for nd in n3_dots:
        Tmc, Q, Teff = solve_Tmc(Ti_fixed, nd, UA)
        Tmc_arr.append(Tmc); Q_arr.append(Q); Teff_arr.append(Teff)
    results_UA[UA] = (np.array(Tmc_arr), np.array(Q_arr), np.array(Teff_arr))




fig1, ax = plt.subplots()
for Ti in T_inlets:
    Tmc, _, _ = results_Tinlet[Ti]
    mask = ~np.isnan(Tmc)
    ax.plot(n3_dots[mask]*1e3, Tmc[mask]*1e3, label=f'{Ti*1e3:.0f} mK')
ax.set_xscale('log')
ax.set_xlabel('He-3 Flow Rate [mmol/s]')
ax.set_ylabel('$T_{mc}$ [mK]')
ax.set_title(f'Mixing Chamber Temp vs Flow Rate\n(varying inlet temp, UA={UA_fixed*1e3:.0f} mW/K, Q_ext={Q_EXT*1e6:.0f} μW)')
ax.legend(title='$T_{inlet}$')
ax.grid(True)
fig1.tight_layout()


fig3, ax = plt.subplots()
for UA in UA_values:
    Tmc, _, _ = results_UA[UA]
    mask = ~np.isnan(Tmc)
    ax.plot(n3_dots[mask]*1e3, Tmc[mask]*1e3, label=f'{UA*1e3:.0f} mW/K')
ax.set_xscale('log')
ax.set_xlabel('He-3 Flow Rate [mmol/s]')
ax.set_ylabel('$T_{mc}$ [mK]')
ax.set_title(f'Mixing Chamber Temp vs Flow Rate\n(varying HX conductance, $T_{{in}}$={Ti_fixed} K, Q_ext={Q_EXT*1e6:.0f} μW)')
ax.legend(title='UA')
ax.grid(True)
fig3.tight_layout()



fig5, ax = plt.subplots()
for UA in UA_values:
    Tmc, _, Teff = results_UA[UA]
    mask = ~np.isnan(Tmc)
    line, = ax.plot(n3_dots[mask]*1e3, Teff[mask]*1e3, label=f'{UA*1e3:.0f} mW/K')
    ax.plot(n3_dots[mask]*1e3, Tmc[mask]*1e3,
            color=line.get_color(), lw=1.0, ls='--')
ax.set_xscale('log')
ax.set_xlabel('He-3 Flow Rate [mmol/s]')
ax.set_ylabel('T [mK]')
ax.set_title('Effective MC Temp with Kapitza correction\n(solid = $T_{eff}$, dashed = $T_{mc}$)')
ax.legend(title='UA')
ax.grid(True)
fig5.tight_layout()

