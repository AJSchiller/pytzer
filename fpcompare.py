from copy import deepcopy
import numpy as np
import pytzer as pz

# Import Clegg's parameters from the .Rs1 file
rsfile = 'testfiles/FastPitz_MIAMI_25.Rs1'
betaC_rs = np.genfromtxt(rsfile, skip_header=37, max_rows=81, usecols=range(5))
betaCions = np.genfromtxt(rsfile, skip_header=37, max_rows=81, usecols=(5, 6),
    dtype=str)
ao_rs = np.genfromtxt(rsfile, skip_header=122, max_rows=81, usecols=range(3))
thetapsiCCA_rs = np.genfromtxt(rsfile, skip_header=211, max_rows=35,
    delimiter=13, usecols=(0, 1))
thetapsiCCAions = np.genfromtxt(rsfile, skip_header=211, max_rows=35,
    delimiter=(74, 12, 12, 12), usecols=(1, 2, 3), dtype=str, autostrip=True)
thetapsiCAA_rs = np.genfromtxt(rsfile, skip_header=250, max_rows=35,
    delimiter=13, usecols=(0, 1))
thetapsiCAAions = np.genfromtxt(rsfile, skip_header=250, max_rows=35,
    delimiter=(74, 12, 12, 12), usecols=(1, 2, 3), dtype=str, autostrip=True)
lambdaNC_rs = np.genfromtxt(rsfile, skip_header=293, max_rows=6, usecols=0)
lambdaNCions = np.genfromtxt(rsfile, skip_header=293, max_rows=6,
    delimiter=(74, 12, 12), usecols=(1, 2), autostrip=True, dtype=str)
lambdaNA_rs = np.genfromtxt(rsfile, skip_header=302, max_rows=4, usecols=0)
lambdaNAions = np.genfromtxt(rsfile, skip_header=302, max_rows=4,
    delimiter=(74, 12, 12), usecols=(1, 2), autostrip=True, dtype=str)
zetaNCA_rs = np.genfromtxt(rsfile, skip_header=314, max_rows=8, usecols=0)
zetaNCAions = np.genfromtxt(rsfile, skip_header=314, max_rows=8,
    delimiter=(74, 12, 12, 12), usecols=(1, 2, 3), autostrip=True, dtype=str)
rs2ions = {
    'BR': 'Br',
    'B(OH)4': 'BOH4',
    'CL': 'Cl',
    'NA': 'Na',
    'MG': 'Mg',
    'CA': 'Ca',
    'MGOH': 'MgOH',
    'SR': 'Sr',
    'MGF': 'MgF',
    'CAF': 'CaF',
    'CO2*': 'CO2',
    'B(OH)3': 'BOH3',
}
def rs2ion(rs):
    if rs in rs2ions:
        rs = rs2ions[rs]
    return rs

# Set up Pytzer's MIAMI library
tempK = np.array([298.15])
pres = np.array([10.10325]) 
prmlib = deepcopy(pz.libraries.MIAMI)
prmlib.add_zeros(['H', 'Na', 'Mg', 'Ca', 'K', 'MgOH', 'Sr', 'MgF', 'CaF',
    'Cl', 'SO4', 'HSO4', 'OH', 'Br', 'HCO3', 'CO3', 'BOH4', 'F',
    'BOH3', 'CO2', 'HF', 'MgCO3', 'CaCO3', 'SrCO3'])
    
# Tweak values to test differences
#prmlib.bC['H-SO4'] = pz.parameters.bC_H_SO4_CRP94
#prmlib.bC['H-HSO4'] = pz.parameters.bC_H_HSO4_CRP94

# Get corresponding values from Pytzer's MIAMI library
betaC_pz = np.full_like(betaC_rs, np.nan)
ao_pz = np.full_like(ao_rs, np.nan)
for i in range(len(betaCions)):
    ibC = prmlib.bC['-'.join((
        rs2ion(betaCions[i, 0]),
        rs2ion(betaCions[i, 1]),
    ))](tempK, pres)
    betaC_pz[i] = ibC[:5]
    ao_pz[i] = ibC[5:8]
ao_pz[ao_pz == -9] = 0
thetapsiCCA_pz = np.full_like(thetapsiCCA_rs, np.nan)
for i in range(len(thetapsiCCAions)):
    if thetapsiCCAions[i, 2] == '':
        ions = [rs2ion(thetapsiCCAions[i, 0]), rs2ion(thetapsiCCAions[i, 1])]
        ions.sort()
        thetapsiCCA_pz[i, 0] = prmlib.theta['-'.join(ions)](tempK, pres)[0]
    else:
        cations = [
            rs2ion(thetapsiCCAions[i, 0]),
            rs2ion(thetapsiCCAions[i, 1]),
        ]
        cations.sort()
        ions = [*cations, rs2ion(thetapsiCCAions[i, 2])]
        thetapsiCCA_pz[i, 1] = prmlib.psi['-'.join(ions)](tempK, pres)[0]
thetapsiCAA_pz = np.full_like(thetapsiCAA_rs, np.nan)
for i in range(len(thetapsiCAAions)):
    if thetapsiCAAions[i, 2] == '':
        ions = [rs2ion(thetapsiCAAions[i, 0]), rs2ion(thetapsiCAAions[i, 1])]
        ions.sort()
        thetapsiCAA_pz[i, 0] = prmlib.theta['-'.join(ions)](tempK, pres)[0]
    else:
        anions = [
            rs2ion(thetapsiCAAions[i, 0]),
            rs2ion(thetapsiCAAions[i, 1]),
        ]
        anions.sort()
        ions = [rs2ion(thetapsiCAAions[i, 2]), *anions]
        thetapsiCAA_pz[i, 1] = prmlib.psi['-'.join(ions)](tempK, pres)[0]
lambdaNC_pz = np.full_like(lambdaNC_rs, np.nan)
for i in range(len(lambdaNCions)):
    ions = [rs2ion(lambdaNCions[i, 0]), rs2ion(lambdaNCions[i, 1])]
    lambdaNC_pz[i] = prmlib.lambd['-'.join(ions)](tempK, pres)[0]
lambdaNA_pz = np.full_like(lambdaNA_rs, np.nan)
for i in range(len(lambdaNAions)):
    ions = [rs2ion(lambdaNAions[i, 0]), rs2ion(lambdaNAions[i, 1])]
    lambdaNA_pz[i] = prmlib.lambd['-'.join(ions)](tempK, pres)[0]
zetaNCA_pz = np.full_like(zetaNCA_rs, np.nan)
for i in range(len(zetaNCAions)):
    ions = [
        rs2ion(zetaNCAions[i, 0]),
        rs2ion(zetaNCAions[i, 1]),
        rs2ion(zetaNCAions[i, 2]),
    ]
    zetaNCA_pz[i] = prmlib.zeta['-'.join(ions)](tempK, pres)[0]

# Evaluate and print out differences
def sigfig(x, sig):
    return np.round(x, decimals=sig-np.int(np.ceil(np.log10(np.abs(x)))))
def sigfig1(vec):
    for i in range(len(vec)):
        if vec[i] != 0 and not np.isnan(vec[i]):
                vec[i] = sigfig(vec[i], 6)
    return vec
def sigfig2(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i, j] != 0 and not np.isnan(arr[i, j]):
                arr[i, j] = sigfig(arr[i, j], 6)
    return arr
betaC_pz = sigfig2(betaC_pz)
thetapsiCCA_pz = sigfig2(thetapsiCCA_pz)
thetapsiCAA_pz = sigfig2(thetapsiCAA_pz)
lambdaNC_pz = sigfig1(lambdaNC_pz)
lambdaNA_pz = sigfig1(lambdaNA_pz)
zetaNCA_pz = sigfig1(zetaNCA_pz)
d_betaC = betaC_pz - betaC_rs
d_ao = ao_pz - ao_rs
d_thetapsiCCA = thetapsiCCA_pz - thetapsiCCA_rs
d_thetapsiCAA = thetapsiCAA_pz - thetapsiCAA_rs
d_lambdaNC = lambdaNC_pz - lambdaNC_rs
d_lambdaNA = lambdaNA_pz - lambdaNA_rs
d_zetaNCA = zetaNCA_pz - zetaNCA_rs
with open('testfiles/fpcompare.txt', 'w') as f:
    f.write(('{:6} {:6}' + '{:^14}'*5 + '\n').format(
        'cat', 'ani', 'beta0', 'beta1', 'beta2', 'C0', 'C1'))
    for i, ions in enumerate(betaCions):
        if any(np.abs(d_betaC[i]) > 0):
            f.write(('{:6} {:6}' + '{:>14.6e}'*5 + '\n').format(
                *ions, *d_betaC[i]))
    f.write(('\n{:6} {:6} ' + '{:^6}'*3 + '\n').format(
        'cat', 'ani', 'a0', 'a1', 'om'))
    for i, ions in enumerate(betaCions):
        if any(np.abs(d_ao[i]) > 0):
            f.write(('{:6} {:6}' + '{:>6.1f}'*3 + '\n').format(
                *ions, *d_ao[i]))
    f.write('\n{:6} {:6} {:6}{:^14}{:^14}\n'.format(
        'cat1', 'cat2', 'ani', 'theta', 'psi'))
    for i, ions in enumerate(thetapsiCCAions):
        if np.abs(d_thetapsiCCA[i, 0]) > 0:
            f.write('{:6} {:6} {:6}{:>14.6e}\n'.format(
                *ions, d_thetapsiCCA[i, 0]))
        elif np.abs(d_thetapsiCCA[i, 1]) > 0:
            f.write('{:6} {:6} {:6}{:14}{:>14.6e}\n'.format(
                *ions, '', d_thetapsiCCA[i, 1]))
    f.write('\n{:6} {:6} {:6}{:^14}{:^14}\n'.format(
        'ani1', 'ani2', 'cat', 'theta', 'psi'))
    for i, ions in enumerate(thetapsiCAAions):
        if np.abs(d_thetapsiCAA[i, 0]) > 0:
            f.write('{:6} {:6} {:6}{:>14.6e}\n'.format(
                *ions, d_thetapsiCAA[i, 0]))
        elif np.abs(d_thetapsiCAA[i, 1]) > 0:
            f.write('{:6} {:6} {:6}{:14}{:>14.6e}\n'.format(
                *ions, '', d_thetapsiCAA[i, 1]))
    f.write('\n{:6} {:6}{:^14}\n'.format('neut', 'ion', 'lambda'))
    for i, ions in enumerate(lambdaNCions):
        if np.abs(d_lambdaNC[i]) > 0:
            f.write('{:6} {:6}{:>14.6e}\n'.format(*ions, d_lambdaNC[i]))
    for i, ions in enumerate(lambdaNAions):
        if np.abs(d_lambdaNA[i]) > 0:
            f.write('{:6} {:6}{:>14.6e}\n'.format(*ions, d_lambdaNA[i]))
    f.write('\n{:6} {:6} {:6}{:^14}\n'.format('neut', 'cat', 'ani', 'zeta'))
    for i, ions in enumerate(zetaNCAions):
        if np.abs(d_zetaNCA[i]) > 0:
            f.write('{:6} {:6} {:6}{:>14.6e}\n'.format(*ions, d_zetaNCA[i]))
