from copy import deepcopy
import pytzer as pz
from pytzer.equilibrate import _sig01
from autograd.numpy import array
from autograd import elementwise_grad as egrad

# Import dataset
filename = 'testfiles/MilleroStandard.csv'
tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(filename)
tots = tots.ravel()
fixmols = fixmols.ravel()
allions = pz.properties.getallions(eles, fixions)
prmlib = deepcopy(pz.libraries.MIAMI)
prmlib.add_zeros(allions) # just in case
tH2CO3 = tots[eles == 't_H2CO3']
tMg = tots[eles == 't_Mg']
tCa = tots[eles == 't_Ca']
tSr = tots[eles == 't_Sr']
tF = tots[eles == 't_F']

# Auto-prepare eqstate and equilibria lists
_serieses = {
    't_H2CO3': ('CO2', 'HCO3', 'CO3'),
    't_HSO4': ('HSO4', 'SO4'),
    't_F': ('HF', 'F'),
}
_parasites = {
    't_Ca': {
        'CaCO3': (('CaCO3',), 'XCO3'),
        'CaF': (('CaF',), 'XF'),
    },
    't_Mg': {
        'MgCO3': (('MgCO3',), 'XCO3'),
        'MgF': (('MgF',), 'XF'),
        'MgOH': (('MgOH',), 'MgOH'),
    },
    't_Sr': {
        'SrCO3': (('SrCO3',), ('XCO3')),
    },
}
eqions = ['H', 'OH']
equilibria = []
eqtype = []
eqtots = []
for ele in eles:
    if ele in _serieses:
        eqions.extend(_serieses[ele])
        equilibria.extend(_serieses[ele][:-1])
        eqtype.extend(_serieses[ele][:-1])
        eqtots.extend([ele for _ in _serieses[ele][:-1]])
    elif ele in _parasites:
        for parasite in _parasites[ele]:
            if parasite in prmlib.lnk:
                eqions.extend(_parasites[ele][parasite][0])
                equilibria.append(parasite)
                eqtype.append(_parasites[ele][parasite][1])
                eqtots.append(ele)
eqstate = [0.0 for _ in equilibria]
if 'H2O' in prmlib.lnk:
    eqstate.append(30.0)
    equilibria.append('H2O')
    eqtype.append('H2O')
    eqtots.append('H2O')
eqXCO3 = [equilibria[q] for q, eqt in enumerate(eqtype) if eqt == 'XCO3']
eqXF = [equilibria[q] for q, eqt in enumerate(eqtype) if eqt == 'XF']
eqindex = equilibria.index

def _getfractions(eqstate, eqixs):
    if len(eqixs) == 3:
        fXaY = _sig01(eqstate[eqixs[0]])
        fXbY = _sig01(eqstate[eqixs[1]])*(1.0 - fXaY)
        fXcY = 1.0 - (fXaY + fXbY)
        fXYs = [fXaY, fXbY, fXcY]
    elif len(eqixs) == 2:
        fXaY = _sig01(eqstate[eqixs[0]])
        fXbY = 1.0 - fXaY
        fXYs = [fXaY, fXbY]
    elif len(eqixs) == 1:
        fXYs = [1.0]
    return fXYs

def _getXYs(eqstate, eqixs, tots, tots_taken, eles, tY):
    fXYs = _getfractions(eqstate, eqixs)
    max_XYs = [tots[eles == eqtots[eqix]] - tots_taken[eles == eqtots[eqix]]
        for eqix in eqixs]
    max_XY_mults = [(max_XYs[q]/(fXYs[q]*tY))[0] for q in range(len(eqixs))]
    max_XY_mults.append(1.0)
    max_XY_mult = min(max_XY_mults)
    max_XY = tY*max_XY_mult
    fXY = _sig01(eqstate[eqixs[-1]])
    tXY = max_XY*fXY
    mXYs = [fXY*tXY for fXY in fXYs]
    return mXYs

# Extract info from eqstate - carbonates
def getXCO3(eqstate, tots, tots_taken, eles, eqindex, eqXCO3, eqtots, tH2CO3):
    """Calculate X-CO3 concentrations from eqstate."""
    eqixs = [eqindex(eq) for eq in eqXCO3]
    mXCO3s = _getXYs(eqstate, eqixs, tots, tots_taken, eles, tH2CO3)
    for q, eq in enumerate(eqXCO3):
        if eq == 'MgCO3':
            mMgCO3 = mXCO3s[q]
        elif eq == 'CaCO3':
            mCaCO3 = mXCO3s[q]
        elif eq == 'SrCO3':
            mSrCO3 = mXCO3s[q]
    if 'MgCO3' not in eqXCO3:
        mMgCO3 = 0.0
    if 'CaCO3' not in eqXCO3:
        mCaCO3 = 0.0
    if 'SrCO3' not in eqXCO3:
        mSrCO3 = 0.0
    return mMgCO3, mCaCO3, mSrCO3

def getXF(eqstate, tots, tots_taken, eles, eqindex, eqXF, eqtots, tF):
    """Calculate X-F concentrations from eqstate."""
    eqixs = [eqindex(eq) for eq in eqXF]
    mXFs = _getXYs(eqstate, eqixs, tots, tots_taken, eles, tF)
    for q, eq in enumerate(eqXF):
        if eq == 'MgF':
            mMgF = mXFs[q]
        elif eq == 'CaF':
            mCaF = mXFs[q]
    if 'MgF' not in eqXF:
        mMgF = 0.0
    if 'CaF' not in eqXF:
        mCaF = 0.0
    return mMgF, mCaF

def getallXY(eqstate, tots, eles, eqindex, eqXCO3, eqXF, eqtots, tH2CO3, tF):
    tots_taken = array([0.0 for _ in tots])
    mMgCO3, mCaCO3, mSrCO3 = getXCO3(eqstate, tots, tots_taken, eles, eqindex,
        eqXCO3, eqtots, tH2CO3)
    taken = {
    't_Mg': mMgCO3,
    't_Ca': mCaCO3,
    't_Sr': mSrCO3,
    }
    tots_taken = array([taken[ele][0] if ele in taken else 0.0
        for ele in eles])
    mMgF, mCaF = getXF(eqstate, tots, tots_taken, eles, eqindex, eqXF, eqtots,
        tF)
    return mMgCO3, mCaCO3, mSrCO3, mMgF, mCaF

# Can we grad it? Yes we can!
getMgF = (lambda eqstate, tots, eles, eqindex, eqXCO3, eqXF, eqtots, tH2CO3, tF:
    getallXY(eqstate, tots, eles, eqindex, eqXCO3, eqXF, eqtots, tH2CO3, tF)[3])
mMgFdir = getMgF(eqstate, tots, eles, eqindex, eqXCO3, eqXF, eqtots, tH2CO3, tF)
getgrad = egrad(getMgF)
mMgFgrad = getgrad(eqstate, tots, eles, eqindex, eqXCO3, eqXF, eqtots, tH2CO3, tF)

## Given that, what's going on with XF?
#fMgF = pz.equilibrate._sig01(sigXCO3[3])
#fCaF = 1.0 - fMgF
#max_MgF_mult = (tMg - mMgCO3)/(fMgF*tF)
#max_CaF_mult = (tCa - mCaCO3)/(fCaF*tF)
#max_XF_mult = min((1.0, max_MgF_mult, max_CaF_mult))
#max_XF = tF*max_XF_mult
#tXF = pz.equilibrate._sig01(sigXCO3[4])*max_XF
#mMgF = tXF*fMgF
#mCaF = tXF*fCaF
#
## Get final metal totals
#mMg = tMg - (mMgCO3 + mMgF)
#mCa = tCa - (mCaCO3 + mCaF)
#mSr = tSr - mSrCO3
