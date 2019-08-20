from autograd import numpy as np
from scipy.optimize import Bounds, minimize#, LinearConstraint
import pytzer as pz

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

eles2ions = {
    # Acid-base serieses
    't_BOH3': {
        'BOH3': ('BOH3', 'BOH4'),
    },
    't_F': {
        'HF': ('HF', 'F'),
    },
    't_H2CO3': {
        'H2CO3': ('CO2', 'HCO3', 'CO3'),
    },
    't_HSO4': {
        'HSO4': ('HSO4', 'SO4'),
    },
    # Metal parasites
    't_Ca': {
        'CaCO3': ('Ca', 'CaCO3'),
        'CaF': ('Ca', 'CaF'),
    },
    't_Mg': {
        'MgCO3': ('Mg', 'MgCO3'),
        'MgF': ('Mg', 'MgF'),
        'MgOH': ('Mg', 'MgOH'),
    },
    't_Sr': {
        'SrF': ('Sr', 'SrF'),
    },
}

def getvarions(eles, prmlib):
    """Get variable-molality ions with equilibrium constants in the parameter
    library from list of eles.
    """
    varions = ['H', 'OH']
    for ele in eles:
        for eq in eles2ions[ele]:
            if eq in prmlib.lnk:
                for ion in eles2ions[ele][eq]:
                    if ion not in varions:
                        varions.append(ion)
    return tuple(varions)

def getvarmolsD(eqstate, tots1, eles, fixZB):
    varmolsD = {}
    varZB = 0.0
    if 't_HSO4' in eles:
        tHSO4 = tots1[eles.index('t_HSO4')]
        varmolsD['HSO4'] = eqstate[1]
        varmolsD['SO4'] = tHSO4 - varmolsD['HSO4']
        varZB = varZB - varmolsD['HSO4'] - 2*varmolsD['SO4']
    mH_zb, mOH_zb = _mH_mOH_from_zb(fixZB, varZB)
    pH2O_dissoc = eqstate[0]
    mH2O_dissoc = 10.0**-pH2O_dissoc
    varmolsD['H'] = mH_zb + mH2O_dissoc
    varmolsD['OH'] = mOH_zb + mH2O_dissoc
    return varmolsD

def getallmolsT(eqstate, tots1, eles, fixmolsT, fixZB):
    varmolsD = getvarmolsD(eqstate, tots1, eles, fixZB)
    varions = list(varmolsD.keys())
    allmolsT = np.array([np.append(fixmolsT,
        [varmolsD[ion] for ion in varions])])
    allions = [ion for ion in fixions]
    allions.extend(varions)
    return allmolsT, allions

def getGibbsComponents(eqstate, fixmolsT, fixZB, varions, allindex,
        allmxs, tots1, eles):
    allmolsT, allions = getallmolsT(eqstate, tots1, eles, fixmolsT, fixZB)
    lnaw = pz.matrix.lnaw(allmolsT, allmxs)
    lnacfs = pz.matrix.ln_acfs(allmolsT, allmxs)
    allindex = allions.index
    gargs = (allindex, allmolsT, lnacfs)
    mH, lnacfH = _get_m_lnacf('H', *gargs)
    mOH, lnacfOH = _get_m_lnacf('OH', *gargs)
    mHSO4, lnacfHSO4 = _get_m_lnacf('HSO4', *gargs)
    mSO4, lnacfSO4 = _get_m_lnacf('SO4', *gargs)
    GH2O = pz.equilibrate._GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O)[0]
    GHSO4 = pz.equilibrate._GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4,
        lnacfHSO4, lnkHSO4)[0]
    return GH2O, GHSO4
    
def getGibbs(eqstate, fixmolsT, fixchargesT, varions, allindex, allmxs, tots1,
        eles):
    GH2O, GHSO4 = getGibbsComponents(eqstate, fixmolsT, fixchargesT, varions,
        allindex, allmxs, tots1, eles)
    return GH2O**2 + GHSO4**2

# Define solution conditions
fixmolsT = np.array([[1.0, 1.0]])
fixions = ('Na', 'Cl')
tempK = np.array([298.15])
pres = np.array([10.10325])
prmlib = pz.libraries.MIAMI
eles = ('t_HSO4', 't_BOH3', 't_H2CO3', 't_Ca',)
tots1 = np.array([2.0])

varions = getvarions(eles, prmlib)

# Evaluate thermodynamic equilibrium constants etc.
lnkH2O = prmlib.lnk['H2O'](tempK)
lnkHSO4 = prmlib.lnk['HSO4'](tempK)

## Get ionic charges
#fixchargesT = np.transpose(pz.properties.charges(fixions)[0])
#if len(fixchargesT) == 0:
#    fixZB = 0.0
#else:
#    fixZB = np.sum(fixmolsT*fixchargesT)
#    
#prmlib.add_zeros(allions)
#allindex = allions.index
#allmxs = pz.matrix.assemble(allions, tempK, pres, prmlib=prmlib)
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
