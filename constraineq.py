from autograd import numpy as np
#from autograd import elementwise_grad as egrad
import pytzer as pz

# Define solution conditions
fixmolsT = np.array([[1.0, 1.0]])
fixions = ('Na', 'Cl')
tempK = np.array([298.15])
pres = np.array([10.10325])
prmlib = pz.libraries.MIAMI
eles = ('t_HSO4',)
tots1 = np.array([1.0])

# Evaluate thermodynamic equilibrium constants etc.
lnkH2O = prmlib.lnk['H2O'](tempK)
lnkHSO4 = prmlib.lnk['HSO4'](tempK)

# Get ionic charges
fixchargesT = np.transpose(pz.properties.charges(fixions)[0])

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

allions = [ion for ion in fixions]
varions = ('H', 'OH', 'HSO4', 'SO4')
allions.extend(varions)
allindex = allions.index
allmxs = pz.matrix.assemble(allions, tempK, pres, prmlib=prmlib)

def getGibbsComponents(eqstate, fixmolsT, fixZB, varions, allindex,
        allmxs, tHSO4):
    varmols = {}
    aHSO4 = eqstate[1]
    varmols['HSO4'] = aHSO4*tHSO4
    varmols['SO4'] = tHSO4 - varmols['HSO4']
    zbHSO4 = -(varmols['HSO4'] + 2*varmols['SO4'])
    varZB = zbHSO4
    mH_zb, mOH_zb = _mH_mOH_from_zb(fixZB, varZB)
    pH2O_dissoc = eqstate[0]
    mH2O_dissoc = 10.0**-pH2O_dissoc
    varmols['H'] = mH_zb + mH2O_dissoc
    varmols['OH'] = mOH_zb + mH2O_dissoc
    allmolsT = np.array([np.append(fixmolsT,
        [varmols[ion] for ion in varions])])
    lnaw = pz.matrix.lnaw(allmolsT, allmxs)
    lnacfs = pz.matrix.ln_acfs(allmolsT, allmxs)
    gargs = (allindex, allmolsT, lnacfs)
    mH, lnacfH = _get_m_lnacf('H', *gargs)
    mOH, lnacfOH = _get_m_lnacf('OH', *gargs)
    mHSO4, lnacfHSO4 = _get_m_lnacf('HSO4', *gargs)
    mSO4, lnacfSO4 = _get_m_lnacf('SO4', *gargs)
    GH2O = pz.equilibrate._GibbsH2O(lnaw, mH, lnacfH, mOH, lnacfOH, lnkH2O)[0]
    GHSO4 = pz.equilibrate._GibbsHSO4(mH, lnacfH, mSO4, lnacfSO4, mHSO4,
        lnacfHSO4, lnkHSO4)[0]
    return GH2O, GHSO4
    
def getGibbs(eqstate, fixmolsT, fixchargesT, varions, allindex, allmxs, tHSO4):
    GH2O, GHSO4 = getGibbsComponents(eqstate, fixmolsT, fixchargesT, varions,
        allindex, allmxs, tHSO4)
    return GH2O**2 + GHSO4**2

if len(fixchargesT) == 0:
    fixZB = 0.0
else:
    fixZB = np.sum(fixmolsT*fixchargesT)

eqstate_guess = [13.0, 0.5]
Gargs = (fixmolsT, fixZB, varions, allindex, allmxs, tots1[0])
Gibbs_guess = getGibbs(eqstate_guess, *Gargs)

from scipy.optimize import Bounds, minimize#, LinearConstraint
bounds = Bounds(
    [-np.inf, 0.0],
    [ np.inf, 1.0],
)

# Solve
eqstate = minimize(lambda eqstate: getGibbs(eqstate, *Gargs),
    eqstate_guess, method='trust-constr', bounds=bounds)

# Test solution
pH2O_dissoc = eqstate['x'][0]
Gibbs = getGibbs(eqstate['x'], *Gargs)
