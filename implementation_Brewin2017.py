import numpy as np
import xarray as xr

# -----------------------------------------------------------------------------
# Full Python pseudo-code of Brewin et al. (2017) size-resolved PP model
# -----------------------------------------------------------------------------
# Includes:
# - Eq. (3): Zp(Bs)
# - Eq. (5): B(z)
# - Eq. (6)–(9): Size partition B1,B2,B3
# - Eq. (13)–(14): PBm,i(z), αB,i(z)
# - Eq. (15)–(19): Irradiance field I(z,t)
# -----------------------------------------------------------------------------

params = {
    # Chlorophyll partition coefficients (Table 2)
    'Bm12': 1.20,
    'Bm1': 0.60,
    'S12': 0.70,
    'S1': 1.20,

    # Vertical chlorophyll profile (Eq. 5)
    'SBs': 0.32,
    'E': -0.7,
    'F': -0.2,
    'G': -0.2,
    'H': 0.71,
    'r': 0.29,

    # Photophysiology surface values (Table 2)
    'PBm_s': [4.92, 7.50, 8.62],   # mg C (mg Chl)^-1 h^-1 (pico, nano, micro)
    'alpha_s': [0.040, 0.051, 0.062],

    # Depth decay factors
    'SP': [0.018, 0.012, 0.009],
    'Sa': [0.010, 0.007, 0.006],

    # Optical constant Kc (clear water)
    'Kc': 0.04
}

# -------------------------------
# 1. Euphotic depth (Eq. 3)
# -------------------------------
def euphotic_depth(Bs):
    # Re-tuned Brewin 2017 coefficients (Table 2)
    qa, qb, qc, qd = 1.52, -0.4, 0.0, 0.01
    logZp = qa + qb*np.log10(Bs + qd) + qc*(np.log10(Bs + qd))**2
    return 10**logZp

# -------------------------------
# 2. Vertical chlorophyll profile (Eq. 5)
# -------------------------------
def chlorophyll_profile(Bs, Zp, Zm):
    z = np.linspace(0, 1.5*Zp, 100)
    f = z / Zp
    BBs_m = 10**(params['E']*np.log10(Bs) + params['F'])
    fm = params['G']*np.log10(Bs) + params['H']
    BBs_f = (1 - params['SBs']*f + BBs_m * np.exp(-((f - fm)/params['r'])**2))

    ratio = Zp/Zm
    if ratio < 1.0:
        Bz = np.ones_like(z) * Bs
    elif ratio > 1.5:
        Bz = BBs_f * Bs
    else:
        n = (ratio - 1.0) / 0.5
        Bz = n*(BBs_f*Bs) + (1-n)*Bs
    return z, Bz

# -------------------------------
# 3. Size partition (Eq. 6–9)
# -------------------------------
def size_partition(B):
    B12 = params['Bm12'] * (1 - np.exp(-params['S12']*B))
    B1 = params['Bm1'] * (1 - np.exp(-params['S1']*B))
    B2 = B12 - B1
    B3 = B - B12
    return B1, B2, B3

# -------------------------------
# 4. Photophysiology (Eq. 13–14)
# -------------------------------
def photophysiology(z, Zp, i):
    f = z / Zp
    PBm = params['PBm_s'][i] * np.exp(-params['SP'][i]*f)
    alpha = params['alpha_s'][i] * np.exp(-params['Sa'][i]*f)
    return PBm, alpha

# -------------------------------
# 5. Irradiance (Eq. 15–19)
# -------------------------------
def irradiance_profile(PAR, Zp, z, lat=0, doy=180):
    # Daylength D (hours), simplified as 12 h (needs astronomical function)
    D = 12.0
    PAR_umol = PAR * 1e6
    Im0_plus = (PAR_umol / (2*D)) * np.pi  # Eq. 15
    Im0_minus = Im0_plus * 0.98  # 2% reflection

    hours = np.arange(1, int(D)+1)
    Izt = []
    for t in hours:
        I0t = Im0_minus * np.sin(np.pi * t / D) / 3600.0  # Eq. 16
        # Diffuse attenuation coefficient (Eq. 18–19)
        KZp = 4.6 / Zp
        Bz_dummy = np.ones_like(z)
        Kv = (KZp - params['Kc']) * (Bz_dummy / np.mean(Bz_dummy)) + params['Kc']
        Iz = I0t * np.exp(-Kv * z)
        Izt.append(Iz)
    return hours, np.array(Izt)

# -------------------------------
# 6. Production profile (Eq. 2)
# -------------------------------
def production_profile(Bz, Iz_t, z, Zp):
    B1, B2, B3 = size_partition(Bz)
    B_all = [B1, B2, B3]
    P_classes = [0.0, 0.0, 0.0]
    
    for i in range(3):
        PBm, alpha = photophysiology(z, Zp, i)
        for Iz in Iz_t:
            Pz = B_all[i] * PBm * (1 - np.exp(-alpha * Iz / PBm))
            P_classes[i] += np.trapz(Pz, z)
    return P_classes

# -------------------------------
# 7. Main driver
# -------------------------------
def compute_primary_production(Bs, PAR, Zm):
    Zp = euphotic_depth(Bs)
    z, Bz = chlorophyll_profile(Bs, Zp, Zm)
    hours, Iz_t = irradiance_profile(PAR, Zp, z)

    P_classes = production_profile(Bz, Iz_t, z, Zp)
    P_total = sum(P_classes)
    return P_classes, P_total

# -------------------------------
# 8. Loop over DataArray grid
# -------------------------------
def run_algorithm(data_path):
    """
    Output in mg-C.m⁻².yr⁻¹
    """

    ds_chl = xr.open_dataset(data_path+"/chl_2000-2020.nc")
    ds_PAR = xr.open_dataset(data_path+"/PAR_2000-2020.nc")
    ds_mlotst = xr.open_dataset(data_path+"/mlotst_2000-2020.nc")

    P1 = xr.zeros_like(ds_chl.CHL)
    P2 = xr.zeros_like(ds_chl.CHL)
    P3 = xr.zeros_like(ds_chl.CHL)
    PT = xr.zeros_like(ds_chl.CHL)

    for t in tqdm(range(ds_chl.dims['time'])):
        for j in range(ds_chl.dims['lat']):
            for i in range(ds_chl.dims['lon']):
                Bs_val = float(ds_chl.CHL[t,j,i].values)
                PAR_val = float(ds_PAR.PAR[t,j,i].values)
                Zm_val = float(ds_mlotst.mlotst[t,j,i].values)

                P_classes, P_total = compute_primary_production(Bs_val, PAR_val, Zm_val)
                P1[t,j,i], P2[t,j,i], P3[t,j,i] = P_classes
                PT[t,j,i] = P_total

    return xr.Dataset({
        'P1': P1,
        'P2': P2,
        'P3': P3,
        'P_total': PT
    })

# -------------------------------
# 9. Monte Carlo uncertainty module
# -------------------------------
# This module performs Monte Carlo sampling of inputs and model parameters
# to estimate uncertainty in P (standard deviation in log10(P), denoted D
# in Brewin et al. 2017). Default uncertainties are taken from the paper
# where provided, or set to conservative estimates otherwise.

mc_defaults = {
    'N': 200,                        # number of Monte Carlo iterations
    'Bs_log10_sd': 0.16,             # uncertainty in log10(Bs)
    'PAR_rel_sd': 0.07,              # relative uncertainty in PAR (7%)
    'Zm_rel_sd': 0.30,               # relative uncertainty in Zm (30%)
    # parameter uncertainties (from Table 2 where available)
    'PBm_s_sd': [0.40, 0.52, 0.76],  # absolute sd for PBm surface values (mg C mgChl^-1 h^-1)
    'alpha_s_rel_sd': [0.10, 0.10, 0.10], # relative sd for alpha_s (~10%)
    'SP_sd': [0.004, 0.003, 0.002],  # absolute sd for SP
    'Sa_sd': [0.003, 0.002, 0.002],  # absolute sd for Sa
}


def monte_carlo_pixel(Bs0, PAR0, Zm0, N=mc_defaults['N'], rng=None):
    """Run Monte Carlo for single pixel. Returns mean P and D = std(log10(P))."""
    if rng is None:
        rng = np.random.default_rng()

    P_samples = np.zeros(N)
    for k in range(N):
        # perturb inputs
        logBs = np.log10(Bs0) + rng.normal(0, mc_defaults['Bs_log10_sd'])
        Bs_k = 10**logBs
        PAR_k = PAR0 * max(1e-6, 1 + rng.normal(0, mc_defaults['PAR_rel_sd']))
        Zm_k = max(1.0, Zm0 * max(0.01, 1 + rng.normal(0, mc_defaults['Zm_rel_sd'])))

        # perturb parameters
        PBm_k = [max(0.01, rng.normal(params['PBm_s'][i], mc_defaults['PBm_s_sd'][i])) for i in range(3)]
        alpha_k = [max(1e-6, rng.normal(params['alpha_s'][i], params['alpha_s'][i]*mc_defaults['alpha_s_rel_sd'][i])) for i in range(3)]
        SP_k = [max(0.0, rng.normal(params['SP'][i], mc_defaults['SP_sd'][i])) for i in range(3)]
        Sa_k = [max(0.0, rng.normal(params['Sa'][i], mc_defaults['Sa_sd'][i])) for i in range(3)]

        # temporarily override params for this run
        params_backup = {k: params[k] for k in ['PBm_s','alpha_s','SP','Sa']}
        params['PBm_s'] = PBm_k
        params['alpha_s'] = alpha_k
        params['SP'] = SP_k
        params['Sa'] = Sa_k

        # compute
        P_classes, P_total = compute_primary_production(Bs_k, PAR_k, Zm_k)
        P_samples[k] = max(1e-12, P_total)

        # restore params
        params['PBm_s'] = params_backup['PBm_s']
        params['alpha_s'] = params_backup['alpha_s']
        params['SP'] = params_backup['SP']
        params['Sa'] = params_backup['Sa']

    # compute statistics in log10 space as Brewin et al. (2017)
    logP = np.log10(P_samples)
    D = np.std(logP, ddof=1)
    P_mean = np.mean(P_samples)
    P_median = np.median(P_samples)
    # return mean, median and D (std in log10 space)
    return P_mean, P_median, D


def run_algorithm_with_uncertainty(ds, N=200):
    """Run full grid algorithm and compute Monte Carlo uncertainty for each pixel.
    WARNING: this is computationally expensive when run per-pixel. Consider
    vectorised or parallel implementations for large grids.
    """
    P_mean = xr.zeros_like(ds.Bs)
    P_median = xr.zeros_like(ds.Bs)
    D_log10 = xr.zeros_like(ds.Bs)

    rng = np.random.default_rng()
    for t in range(ds.dims['time']):
        for j in range(ds.dims['lat']):
            for i in range(ds.dims['lon']):
                Bs_val = float(ds.Bs[t,j,i].values)
                PAR_val = float(ds.PAR[t,j,i].values)
                Zm_val = float(ds.Zm[t,j,i].values)

                mean_p, med_p, D = monte_carlo_pixel(Bs_val, PAR_val, Zm_val, N=N, rng=rng)
                P_mean[t,j,i] = mean_p
                P_median[t,j,i] = med_p
                D_log10[t,j,i] = D

    return xr.Dataset({'P_mean': P_mean, 'P_median': P_median, 'D_log10': D_log10})
