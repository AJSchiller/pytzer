from autograd import numpy as np
from scipy.optimize import Bounds, minimize#, LinearConstraint
import pytzer as pz

def _checkprmlib(eqs, lnkfuncs):
    for eq in eqs:
        if eq not in lnkfuncs:
            print('WARNING: {} '.format(eq) +
                'equilibrium constant missing from parameter library.')

def get_varions_equilibria(eles, prmlib):
    """Get variable-molality ions with equilibrium constants in the parameter
    library from list of eles, and corresponding list of equilibria.
    """
    lnkfuncs = set(prmlib.lnk)
    varions = ['H', 'OH']
    if 'H2O' in lnkfuncs:
        equilibria = ['H2O']
    else:
        equilibria = []
    if 't_BOH3' in eles:
        varions.extend(('BOH3', 'BOH4'))
        equilibria.append('BOH3')
        _checkprmlib(('BOH3',), lnkfuncs)
    if 't_F' in eles:
        varions.extend(('HF', 'F'))
        equilibria.append('HF')
        _checkprmlib(('HF',), lnkfuncs)
    if 't_H2CO3' in eles:
        varions.extend(('CO2', 'HCO3', 'CO3'))
        equilibria.extend(('H2CO3', 'HCO3'))
        _checkprmlib(('H2CO3', 'HCO3'), lnkfuncs)
    if 't_HSO4' in eles:
        varions.extend(('HSO4', 'SO4'))
        equilibria.append('HSO4')
        _checkprmlib(('HSO4',), lnkfuncs)
    if 't_Ca' in eles:
        varions.append('Ca')
        if 't_F' in eles and 'CaF' in lnkfuncs:
            varions.append('CaF')
            equilibria.append('CaF')
        if 't_H2CO3' in eles and 'CaCO3' in lnkfuncs:
            varions.append('CaCO3')
            equilibria.append('CaCO3')
    if 't_Mg' in eles:
        varions.append('Mg')
        if 't_F' in eles and 'MgF' in lnkfuncs:
            varions.append('MgF')
            equilibria.append('MgF')
        if 't_H2CO3' in eles and 'MgCO3' in lnkfuncs:
            varions.append('MgCO3')
            equilibria.append('MgCO3')
        if 'MgOH' in lnkfuncs:
            varions.append('MgOH')
            equilibria.append('MgOH')
    if 't_Sr' in eles:
        varions.append('Sr')
        if 't_H2CO3' in eles and 'SrCO3' in lnkfuncs:
            varions.append('SrCO3')
            equilibria.append('SrCO3')
    return tuple(varions), tuple(equilibria)

def getlnks(tempK, equilibria, prmlib):
    """Evaluate ln(K) values for listed equilibria."""
    return {eq: prmlib.lnk[eq](tempK)[0] for eq in equilibria}

def _mH_mOH_from_zb(fixZB, varZB):
    """mH and mOH molalities from charge balance (ZB) only."""
    allZB = fixZB + varZB
    mOH_ZB = (allZB + abs(allZB))/2.0
    mH_ZB = mOH_ZB - allZB
    return mH_ZB, mOH_ZB

def _get_m_lnacf(ion, allindex, allmolsT, lnacfs):
    """Extract molality and log(activity coeff.) for a given ion."""
    ix = allindex(ion)
    return allmolsT[0][ix], lnacfs[ix]

def getvarmolsD(eqstate, equilibria, tots, eles, fixZB):
    eqindex = equilibria.index
    eleindex = eles.index
    varmolsD = {}
    varZB = 0.0
    if 'BOH3' in equilibria:
        q = eqindex('BOH3')
        varmolsD['BOH3'] = eqstate[q]
        varmolsD['BOH4'] = tots[eleindex('t_BOH3')] - varmolsD['BOH3']
        varZB = varZB - varmolsD['BOH4']
    if 'H2CO3' in equilibria and 'HCO3' in equilibria:
        q = eqindex('H2CO3')
        varmolsD['CO2'] = eqstate[q]
        q = eqindex('HCO3')
        varmolsD['HCO3'] = eqstate[q]
        varmolsD['CO3'] = (tots[eleindex('t_H2CO3')] - varmolsD['CO2'] -
           varmolsD['HCO3'])
        varZB = varZB - varmolsD['HCO3'] - 2*varmolsD['CO3']
    if 'HF' in equilibria:
        q = eqindex('HF')
        varmolsD['HF'] = eqstate[q]
        varmolsD['F'] = tots[eleindex('t_F')] - varmolsD['HF']
        varZB = varZB - varmolsD['F']
    if 'HSO4' in equilibria:
        q = eqindex('HSO4')
        varmolsD['HSO4'] = eqstate[q]
        varmolsD['SO4'] = tots[eleindex('t_HSO4')] - varmolsD['HSO4']
        varZB = varZB - varmolsD['HSO4'] - 2*varmolsD['SO4']
    mH_zb, mOH_zb = _mH_mOH_from_zb(fixZB, varZB)
    if 'H2O' in equilibria:
        q = eqindex('H2O')
        pH2O_dissoc = eqstate[q]
        mH2O_dissoc = 10.0**-pH2O_dissoc
    else:
        mH2O_dissoc = 0.0
    varmolsD['H'] = mH_zb + mH2O_dissoc
    varmolsD['OH'] = mOH_zb + mH2O_dissoc
    return varmolsD

def getallmolsT(fixmolsT, varmolsD, varions):
    return np.array([np.append(fixmolsT, [varmolsD[ion] for ion in varions])])

def getallions(fixions, varions):
    allions = [ion for ion in fixions]
    allions.extend(varions)
    return allions

def getGibbsComponents(eqstate, equilibria, fixmolsT, tots1, eles, fixZB,
        varions, allindex, allmxs, lnks):
    varmolsD = getvarmolsD(eqstate, equilibria, tots1, eles, fixZB)
    allmolsT = getallmolsT(fixmolsT, varmolsD, varions)
    lnaw = pz.matrix.lnaw(allmolsT, allmxs)[0]
    lnacfs = pz.matrix.ln_acfs(allmolsT, allmxs)
    gargs = (allindex, allmolsT, lnacfs)
    mH, lnacfH = _get_m_lnacf('H', *gargs)
    mOH, lnacfOH = _get_m_lnacf('OH', *gargs)
    mHSO4, lnacfHSO4 = _get_m_lnacf('HSO4', *gargs)
    mSO4, lnacfSO4 = _get_m_lnacf('SO4', *gargs)
    GH2O = pz.equilibrate._GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH,
        lnks['H2O'])
    GHSO4 = pz.equilibrate._GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4,
        lnacfHSO4, lnks['HSO4'])
    return GH2O, GHSO4
    
def getGibbs(eqstate, equilibria, fixmolsT, tots1, eles, fixZB, varions,
        allindex, allmxs, lnks):
    GH2O, GHSO4 = getGibbsComponents(eqstate, equilibria, fixmolsT, tots1,
        eles, fixZB, varions, allindex, allmxs, lnks)
    return GH2O**2 + GHSO4**2


# Define solution conditions
fixmolsT = np.array([[1.0, 1.5]])
fixions = ('Na', 'Cl')
tempK = np.array([298.15])
pres = np.array([10.10325])
prmlib = pz.libraries.MIAMI
eles = ('t_HSO4', 't_BOH3', 't_H2CO3', 't_F',)
tots1 = np.array([2.0, 1.0, 1.0, 0.1])
varions, equilibria = get_varions_equilibria(eles, prmlib)
eqstate = [0.1 for _ in equilibria]
lnks = getlnks(tempK, equilibria, prmlib)
fixchargesT = np.transpose(pz.properties.charges(fixions)[0])
if len(fixchargesT) == 0:
    fixZB = 0.0
else:
    fixZB = np.sum(fixmolsT*fixchargesT)
varmolsD = getvarmolsD(eqstate, equilibria, tots1, eles, fixZB)
allmolsT = getallmolsT(fixmolsT, varmolsD, varions)
allions = getallions(fixions, varions)
prmlib.add_zeros(allions)
allindex = allions.index
allmxs = pz.matrix.assemble(allions, tempK, pres, prmlib=prmlib)
GH2O, GHSO4 = getGibbsComponents(eqstate, equilibria, fixmolsT, tots1, eles,
    fixZB, varions, allindex, allmxs, lnks)
GTotal = getGibbs(eqstate, equilibria, fixmolsT, tots1, eles, fixZB, varions,
    allindex, allmxs, lnks)

# Next: cycle through equilibria to auto-generate appropriate eqstate first
# guesses, bounds and constraints
# Test with serieses
# Then go back and add parasites to getvarmolsD()

## Evaluate thermodynamic equilibrium constants etc.
#lnkH2O = prmlib.lnk['H2O'](tempK)
#lnkHSO4 = prmlib.lnk['HSO4'](tempK)

#eqstate_guess = [13.0, 0.5]
#Gargs = (fixmolsT, fixZB, varions, allindex, allmxs, tots1, eles)
#Gibbs_guess = getGibbs(eqstate_guess, *Gargs)
#
#bounds = Bounds(
#    [-np.inf, 0.0],
#    [ np.inf, tots1[0]],
#)
#eqstate = minimize(lambda eqstate: getGibbs(eqstate, *Gargs),
#    eqstate_guess, method='trust-constr', bounds=bounds)
#
## Test solution
#pH2O_dissoc = eqstate['x'][0]
#Gibbs = getGibbs(eqstate['x'], *Gargs)
