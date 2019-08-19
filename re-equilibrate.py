from copy import deepcopy
import pytzer as pz
from pytzer.equilibrate import _sig01
from pytzer import equilibrate as pq
from autograd.numpy import append, array, concatenate, exp, sqrt
from autograd.numpy import sum as np_sum
from autograd import elementwise_grad as egrad

# Import dataset
filename = 'testfiles/MilleroStandard.csv'
tots, fixmols, eles, fixions, tempK, pres = pz.io.gettots(filename)
tots = tots.ravel()
fixmols1 = fixmols.ravel()
fixcharges = pz.properties.charges(fixions)[0].ravel()
if len(fixcharges) == 0:
    zbfixed = 0.0
else:
    zbfixed = np_sum(fixmols1*fixcharges)
allionsOLD = pz.properties.getallions(eles, fixions)
prmlib = deepcopy(pz.libraries.MIAMI)
prmlib.add_zeros(allionsOLD) # just in case

# Auto-prepare eqstate and equilibria lists
_serieses = {
    't_H2CO3': ('CO2', 'HCO3', 'CO3'),
    't_HSO4': ('HSO4', 'SO4'),
    't_F': ('HF', 'F'),
    't_BOH3': ('BOH3', 'BOH4'),
    't_trisH': ('trisH', 'tris'),
    't_NH4': ('NH4', 'NH3'),
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
def _eqprep(eles, prmlib):
    """Generate default eqstate vector, lists of equilibria names and types,
    and ions/solutes involved in the equilibria.
    """
    eqions = ['H', 'OH']
    equilibria = []
    eqtype = []
    eqtots = []
    lnkfuncs = []
    for ele in eles:
        if ele in _serieses:
            eqions.extend(_serieses[ele])
            equilibria.extend(_serieses[ele][:-1])
            eqtype.extend(_serieses[ele][:-1])
            eqtots.extend([str(ele) for _ in _serieses[ele][:-1]])
            lnkfuncs.extend([prmlib.lnk[ele] if ele in prmlib.lnk else
                lambda tempK, pres: 0.0 for ele in _serieses[ele][:-1]])
        elif ele in _parasites:
            for parasite in _parasites[ele]:
                if parasite in prmlib.lnk:
                    eqions.extend(_parasites[ele][parasite][0])
                    equilibria.append(parasite)
                    eqtype.append(_parasites[ele][parasite][1])
                    eqtots.append(str(ele))
                    lnkfuncs.append(prmlib.lnk[parasite])
            eqions.append(ele.split('t_')[1])
    eqstate = [0.0 for _ in equilibria]
    if 'H2O' in prmlib.lnk:
        equilibria.append('H2O')
        eqtype.append('H2O')
        eqtots.append('H2O')
        eqstate.append(30.0)
        lnkfuncs.append(prmlib.lnk['H2O'])
    eqXCO3 = [equilibria[q] for q, eqt in enumerate(eqtype) if eqt == 'XCO3']
    eqXF = [equilibria[q] for q, eqt in enumerate(eqtype) if eqt == 'XF']
    eqindex = equilibria.index
    return eqstate, equilibria, eqindex, eqtots, eqXCO3, eqXF, eqions, lnkfuncs

#eqstate, equilibria, eqindex, eqtots, eqXCO3, eqXF, eqions, lnkfuncs = \
#    _eqprep(eles, prmlib)

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

def _getXYs(eqstate, eqixs, eqtots, tots, tots_taken, eles, tY):
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

def getXCO3(eqstate, tots, tots_taken, eles, eqindex, eqXCO3, eqtots, tH2CO3):
    """Calculate X-CO3 concentrations from eqstate."""
    eqixs = [eqindex(eq) for eq in eqXCO3]
    mXCO3s = _getXYs(eqstate, eqixs, eqtots, tots, tots_taken, eles, tH2CO3)
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

def getXCO3dict(eqstate, moldict, tots, tots_taken, eles, eqindex, eqXCO3,
        eqtots, tH2CO3):
    """Calculate X-CO3 concentrations from eqstate."""
    eqixs = [eqindex(eq) for eq in eqXCO3]
    mXCO3s = _getXYs(eqstate, eqixs, eqtots, tots, tots_taken, eles, tH2CO3)
    for q, eq in enumerate(eqXCO3):
        moldict[eq] = mXCO3s[q]
    return moldict

def getXF(eqstate, tots, tots_taken, eles, eqindex, eqXF, eqtots, tF):
    """Calculate X-F concentrations from eqstate."""
    eqixs = [eqindex(eq) for eq in eqXF]
    mXFs = _getXYs(eqstate, eqixs, eqtots, tots, tots_taken, eles, tF)
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

# don't use this one, use the dict version below instead
def getallXY(eqstate, tots, eles, eqindex, equilibria, eqXCO3, eqXF, eqtots,
        zbfixed):
    tH2CO3 = tots[eles == 't_H2CO3']
    tHSO4 = tots[eles == 't_HSO4']
    tF = tots[eles == 't_F']
    tMg = tots[eles == 't_Mg']
    tCa = tots[eles == 't_Ca']
    tSr = tots[eles == 't_Sr']
    tBOH3 = tots[eles == 't_BOH3']
    tTrisH = tots[eles == 't_trisH']
    tNH4 = tots[eles == 't_NH4']
    # Carbonates
    if len(eqXCO3) > 0:
        tots_taken = array([0.0 for _ in tots])
        mMgCO3, mCaCO3, mSrCO3 = getXCO3(eqstate, tots, tots_taken, eles,
            eqindex, eqXCO3, eqtots, tH2CO3)
    else:
        mMgCO3 = 0.0
        mCaCO3 = 0.0
        mSrCO3 = 0.0
    # Fluorides
    if len(eqXF) > 0:
        taken = {
            't_Mg': mMgCO3,
            't_Ca': mCaCO3,
        }
        tots_taken = array([taken[ele][0] if ele in taken else 0.0
            for ele in eles])
        mMgF, mCaF = getXF(eqstate, tots, tots_taken, eles, eqindex, eqXF,
            eqtots, tF)
    else:
        mMgF = 0.0
        mCaF = 0.0
    # Magnesium
    if 'MgOH' in equilibria:
        q = eqindex('MgOH')
        mMgOH = (tMg - (mMgCO3 + mMgF))*_sig01(eqstate[q])
    else:
        mMgOH = 0.0
    mMg = tMg - (mMgCO3 + mMgF + mMgOH)
    # Calcium and strontium
    mCa = tCa - (mCaCO3 + mCaF)
    mSr = tSr - mSrCO3
    # Fluoride
    if 'HF' in equilibria:
        q = eqindex('HF')
        mHF = (tF - (mMgF + mCaF))*_sig01(eqstate[q])
    else:
        mHF = 0.0
    mF = tF - (mMgF + mCaF + mHF)
    # Carbonate system
    if 't_H2CO3' in eles:
        tH2CO3rem = tH2CO3 - (mMgCO3 + mCaCO3 + mSrCO3)
        q = eqindex('CO2')
        mCO2 = tH2CO3rem*_sig01(eqstate[q])
        q = eqindex('HCO3')
        mHCO3 = (tH2CO3rem - mCO2)*_sig01(eqstate[q])
        mCO3 = tH2CO3rem - (mCO2 + mHCO3)
    else:
        mCO2 = 0.0
        mHCO3 = 0.0
        mCO3 = 0.0
    # Bisulfate
    if 't_HSO4' in eles:
        q = eqindex('HSO4')
        mHSO4 = tHSO4*_sig01(eqstate[q])
        mSO4 = tHSO4 - mHSO4
    else:
        mHSO4 = 0.0
        mSO4 = 0.0
    # Borate
    if 't_BOH3' in eles:
        q = eqindex('BOH3')
        mBOH3 = tBOH3*_sig01(eqstate[q])
        mBOH4 = tBOH3 - mBOH3
    else:
        mBOH3 = 0.0
        mBOH4 = 0.0
    # TrisH+
    if 't_trisH' in eles:
        q = eqindex('trisH')
        mTrisH = tTrisH*_sig01(eqstate[q])
        mTris = tTrisH - mTrisH
    else:
        mTrisH = 0.0
        mTris = 0.0
    # Ammonium
    if 't_NH4' in eles:
        q = eqindex('NH4')
        mNH4 = tNH4*_sig01(eqstate[q])
        mNH3 = tNH4 - mNH4
    else:
        mNH4 = 0.0
        mNH3 = 0.0
    # Charge balances
    zbHSO4 = -mHSO4 - 2*mSO4
    zbTrisH = mTrisH
    zbH2CO3 = -(mHCO3 + 2*mCO3)
    zbBOH3 = -mBOH4
    zbNH4 = mNH4
    if 't_Mg' in eles:
        zbMg_var = 2*mMg + mMgOH + mMgF
    else:
        zbMg_var = 0.0
    if 't_Ca' in eles:
        zbCa_var = 2*mCa + mCaF
    else:
        zbCa_var = 0.0
    if 't_Sr' in eles:
        zbSr_var = 2*mSr
    else:
        zbSr_var = 0.0
    zbalance = (zbfixed + zbHSO4 + zbTrisH + zbH2CO3 + zbBOH3 + zbMg_var +
        zbCa_var + zbSr_var + zbNH4)
    # Water
    if 'H2O' in equilibria:
        q = eqindex('H2O')
        dissociatedH2O = exp(-eqstate[q])
    else:
        dissociatedH2O = 0.0
    mOH = (zbalance + sqrt(zbalance**2 + dissociatedH2O))/2
    mH = mOH - zbalance
    return (mMgCO3, mCaCO3, mSrCO3, mMgF, mCaF, mMgOH, mMg, mCa, mSr, mHF, mF,
        mCO2, mHCO3, mCO3, mHSO4, mSO4, mBOH3, mBOH4, mTrisH, mTris, mH, mOH,
        mNH4, mNH3)

def _checkgeteles(ele, tots, eles):
    if ele in eles:
        tX = tots[eles == ele]
    else:
        tX = 0.0
    return tX

# ==== Replacement for pytzer.equilibrate._varmols ============================
def getallXYdict(eqstate, tots, eles, eqindex, equilibria, eqXCO3, eqXF,
        eqtots, zbfixed):
    """Calculate all variable molalities from the eqstate."""
    moldict = {}
    tBOH3 = _checkgeteles('t_BOH3', tots, eles)
    tCa = _checkgeteles('t_Ca', tots, eles)
    tH2CO3 = _checkgeteles('t_H2CO3', tots, eles)
    tHSO4 = _checkgeteles('t_HSO4', tots, eles)
    tF = _checkgeteles('t_F', tots, eles)
    tMg = _checkgeteles('t_Mg', tots, eles)
    tSr = _checkgeteles('t_Sr', tots, eles)
    tTrisH = _checkgeteles('t_trisH', tots, eles)
    tNH4 = _checkgeteles('t_NH4', tots, eles)
    # Carbonates
    if len(eqXCO3) > 0:
        tots_taken = array([0.0 for _ in tots])
        moldict = getXCO3dict(eqstate, moldict,
            tots, tots_taken, eles, eqindex, eqXCO3, eqtots, tH2CO3)
    for XCO3 in ['MgCO3', 'CaCO3', 'SrCO3']:
        if XCO3 not in moldict:
            moldict[XCO3] = 0.0
    # Fluorides
    if len(eqXF) > 0:
        taken = {
            't_Mg': moldict['MgCO3'],
            't_Ca': moldict['CaCO3'],
        }
        tots_taken = array([taken[ele][0] if ele in taken else 0.0
            for ele in eles])
        moldict['MgF'], moldict['CaF'] = getXF(eqstate, tots, tots_taken, eles,
            eqindex, eqXF, eqtots, tF)
    else:
        moldict['MgF'] = 0.0
        moldict['CaF'] = 0.0
    # Magnesium
    if 'MgOH' in equilibria:
        q = eqindex('MgOH')
        moldict['MgOH'] = ((tMg - (moldict['MgCO3'] + moldict['MgF']))*
            _sig01(eqstate[q]))
    else:
        moldict['MgOH'] = 0.0
    moldict['Mg'] = tMg - (moldict['MgCO3'] + moldict['MgF'] + moldict['MgOH'])
    # Calcium and strontium
    moldict['Ca'] = tCa - (moldict['CaCO3'] + moldict['CaF'])
    moldict['Sr'] = tSr - moldict['SrCO3']
    # Fluoride
    if 'HF' in equilibria:
        q = eqindex('HF')
        moldict['HF'] = ((tF - (moldict['MgF'] + moldict['CaF']))*
            _sig01(eqstate[q]))
    else:
        moldict['HF'] = 0.0
    moldict['F'] = tF - (moldict['MgF'] + moldict['CaF'] + moldict['HF'])
    # Carbonate system
    if 't_H2CO3' in eles:
        tH2CO3rem = tH2CO3 - (moldict['MgCO3'] + moldict['CaCO3'] +
            moldict['SrCO3'])
        q = eqindex('CO2')
        moldict['CO2'] = tH2CO3rem*_sig01(eqstate[q])
        q = eqindex('HCO3')
        moldict['HCO3'] = (tH2CO3rem - moldict['CO2'])*_sig01(eqstate[q])
        moldict['CO3'] = tH2CO3rem - (moldict['CO2'] + moldict['HCO3'])
    else:
        moldict['CO2'] = 0.0
        moldict['HCO3'] = 0.0
        moldict['CO3'] = 0.0
    # Bisulfate
    if 't_HSO4' in eles:
        q = eqindex('HSO4')
        moldict['HSO4'] = tHSO4*_sig01(eqstate[q])
        moldict['SO4'] = tHSO4 - moldict['HSO4']
    else:
        moldict['HSO4'] = 0.0
        moldict['SO4'] = 0.0
    # Borate
    if 't_BOH3' in eles:
        q = eqindex('BOH3')
        moldict['BOH3'] = tBOH3*_sig01(eqstate[q])
        moldict['BOH4'] = tBOH3 - moldict['BOH3']
    else:
        moldict['BOH3'] = 0.0
        moldict['BOH4'] = 0.0
    # TrisH+
    if 't_trisH' in eles:
        q = eqindex('trisH')
        moldict['trisH'] = tTrisH*_sig01(eqstate[q])
        moldict['tris'] = tTrisH - moldict['trisH']
    else:
        moldict['trisH'] = 0.0
        moldict['tris'] = 0.0
    # Ammonium
    if 't_NH4' in eles:
        q = eqindex('NH4')
        moldict['NH4'] = tNH4*_sig01(eqstate[q])
        moldict['NH3'] = tNH4 - moldict['NH4']
    else:
        moldict['NH4'] = 0.0
        moldict['NH3'] = 0.0
    # Charge balances
    zbHSO4 = -moldict['HSO4'] - 2*moldict['SO4']
    zbTrisH = moldict['trisH']
    zbH2CO3 = -(moldict['HCO3'] + 2*moldict['CO3'])
    zbBOH3 = -moldict['BOH4']
    zbMg_var = 2*moldict['Mg'] + moldict['MgOH'] + moldict['MgF']
    zbCa_var = 2*moldict['Ca'] + moldict['CaF']
    zbSr_var = 2*moldict['Sr']
    zbNH4 = moldict['NH4']
    zbalance = (zbfixed + zbHSO4 + zbTrisH + zbH2CO3 + zbBOH3 + zbMg_var +
        zbCa_var + zbSr_var + zbNH4)
    # Water
    if 'H2O' in equilibria:
        q = eqindex('H2O')
        dissociatedH2O = exp(-eqstate[q])
    else:
        dissociatedH2O = 0.0
    moldict['OH'] = (zbalance + sqrt(zbalance**2 + dissociatedH2O))/2
    moldict['H'] = mOH - zbalance
    return moldict

def _moldict2eqmols(moldict, eqions):
    return concatenate([moldict[ion] for ion in eqions])

# Can we grad it? Yes, we can!
eqstate, equilibria, eqindex, eqtots, eqXCO3, eqXF, eqions, lnkfuncs = \
    _eqprep(eles, prmlib)
xyargs = (tots, eles, eqindex, equilibria, eqXCO3, eqXF, eqtots, zbfixed)
(mMgCO3, mCaCO3, mSrCO3, mMgF, mCaF, mMgOH, mMg, mCa, mSr, mHF, mF,
    mCO2, mHCO3, mCO3, mHSO4, mSO4, mBOH3, mBOH4, mTrisH, mTris, mH, mOH,
    mNH4, mNH3) = \
    getallXY(eqstate, *xyargs)
getgrad = egrad(lambda eqstate, xyargs, i: getallXY(eqstate, *xyargs)[i])
grad_test = getgrad(eqstate, xyargs, 0)
moldict = getallXYdict(eqstate, *xyargs)
dictgrad = egrad(lambda eqstate, xyargs, i: getallXYdict(eqstate, *xyargs)[i])
grad_dict_test = dictgrad(eqstate, xyargs, 'MgCO3')
eqmols = _moldict2eqmols(moldict, eqions)

def geteqmols(eqstate, tots, eles, eqindex, equilibria, eqXCO3, eqXF, eqtots,
        zbfixed, eqions):
    moldict = getallXYdict(eqstate, tots, eles, eqindex, equilibria, eqXCO3,
        eqXF, eqtots, zbfixed)
    eqmols = _moldict2eqmols(moldict, eqions)
    return eqmols

eqmols2 = geteqmols(eqstate, tots, eles, eqindex, equilibria, eqXCO3, eqXF,
    eqtots, zbfixed, eqions)
deqmols2 = egrad(lambda eqstate, eqargs: geteqmols(eqstate, *eqargs)[1])
dtest = deqmols2(eqstate, (tots, eles, eqindex, equilibria, eqXCO3, eqXF,
     eqtots, zbfixed, eqions))

def getallmols(eqstate, tots, eles, eqindex, equilibria, eqXCO3, eqXF, eqtots,
        zbfixed, eqions, fixmols, fixions):
    eqmols = geteqmols(eqstate, tots, eles, eqindex, equilibria, eqXCO3, eqXF,
        eqtots, zbfixed, eqions)
    allmols = append(fixmols.ravel(), eqmols)
    allions = append(fixions, eqions)
    return allmols, allions

allmols, allions = getallmols(eqstate, tots, eles, eqindex, equilibria, eqXCO3,
    eqXF, eqtots, zbfixed, eqions, fixmols, fixions)

dallmols = egrad(lambda eqstate, eqargs: getallmols(eqstate, *eqargs)[0][5])
dalltest = dallmols(eqstate, (tots, eles, eqindex, equilibria, eqXCO3, eqXF,
    eqtots, zbfixed, eqions, fixmols, fixions))

def _eqtotcheck(equilibrium, equilibria, ele, eles, tots1, eqindex, lnks,
        Gfunc, Gargs):
    if equilibrium in equilibria:
        if equilibrium == 'H2O':
            q = eqindex(equilibrium)
            gEq = Gfunc(*Gargs, lnks[q])
        else:
            if tots1[eles == ele] > 0:
                q = eqindex(equilibrium)
                gEq = Gfunc(*Gargs, lnks[q])
            else:
                gEq = 0.0
    else:
        gEq = 0.0
    return gEq

def newGibbs(eqstate, tots, eles, eqindex, equilibria, eqXCO3, eqXF, eqtots,
        zbfixed, eqions, fixmols, fixions, allmxs, lnks, tots1, ideal=False):
    allmols, allions = getallmols(eqstate, tots, eles, eqindex, equilibria,
        eqXCO3, eqXF, eqtots, zbfixed, eqions, fixmols, fixions)
    allmols = array([allmols])
    # Get activities:
    if ideal:
        lnaw = 0.0
        lnacfH = 0.0
        lnacfOH = 0.0
        lnacfHSO4 = 0.0
        lnacfSO4 = 0.0
        lnacfMg = 0.0
        lnacfMgOH = 0.0
        lnacfTris = 0.0
        lnacfTrisH = 0.0
        lnacfCO2 = 0.0
        lnacfHCO3 = 0.0
        lnacfCO3 = 0.0
        lnacfBOH3 = 0.0
        lnacfBOH4 = 0.0
    else:
        lnaw = pz.matrix.lnaw(allmols, allmxs)
        lnacfs = pz.matrix.ln_acfs(allmols, allmxs)
        lnacfH = lnacfs[allions == 'H']
        lnacfOH = lnacfs[allions == 'OH']
        lnacfHSO4 = lnacfs[allions == 'HSO4']
        lnacfSO4 = lnacfs[allions == 'SO4']
        lnacfMg = lnacfs[allions == 'Mg']
        lnacfMgOH = lnacfs[allions == 'MgOH']
        lnacfTris = lnacfs[allions == 'tris']
        lnacfTrisH = lnacfs[allions == 'trisH']
        lnacfCO2 = lnacfs[allions == 'CO2']
        lnacfHCO3 = lnacfs[allions == 'HCO3']
        lnacfCO3 = lnacfs[allions == 'CO3']
        lnacfBOH3 = lnacfs[allions == 'BOH3']
        lnacfBOH4 = lnacfs[allions == 'BOH4']
    # Evaluate equilibrium states:
    eq2eleArgs = {
        'BOH3': ('t_BOH3', pq._GibbsBOH3, (lnaw, lnacfBOH4, mBOH4, lnacfBOH3,
            mBOH3, lnacfH, mH)),
#        'CaCO3': ('t_Ca', pq._GibbsCaCO3, ()),
#        'CaF': ('t_F', pq._GibbsCaF, ()),
        'CO2': ('t_H2CO3', pq._GibbsH2CO3, (lnaw, mH, lnacfH, mHCO3, lnacfHCO3,
            mCO2, lnacfCO2)),
        'H2O': ('', pq._GibbsH2O, (lnaw, mH, lnacfH, mOH, lnacfOH)),
        'HCO3': ('t_H2CO3', pq._GibbsHCO3, (mH, lnacfH, mHCO3, lnacfHCO3, mCO3,
            lnacfCO3)),
#        'HF': ('t_F', pq._GibbsHF, ()),
        'HSO4': ('t_HSO4', pq._GibbsHSO4, (mH, lnacfH, mSO4, lnacfSO4, mHSO4,
            lnacfHSO4)),
#        'MgCO3': ('t_Mg', pq._GibbsMgCO3, ()),
#        'MgF': ('t_F', pq._GibbsMgF, ()),
        'MgOH': ('t_Mg', pq._GibbsMgOH, (mMg, lnacfMg, mMgOH, lnacfMgOH, mOH,
            lnacfOH)),
#        'SrCO3': ('t_Sr', pq._GibbsSrCO3, ()),
        'trisH': ('t_trisH', pq._GibbstrisH, (mH, lnacfH, mTris, lnacfTris,
            mTrisH, lnacfTrisH)),
    }
    GComponents = [_eqtotcheck(eq, equilibria, eq2eleArgs[eq][0], eles, tots1,
        eqindex, lnks, eq2eleArgs[eq][1], eq2eleArgs[eq][2])
        for eq in equilibria if eq in eq2eleArgs]

#    if tots1[eles == 't_H2CO3'] > 0:
#        gH2CO3 = _GibbsH2CO3(lnaw, mH, lnacfH, mHCO3, lnacfHCO3, mCO2,
#            lnacfCO2, lnkH2CO3)
#        gHCO3 = _GibbsHCO3(mH, lnacfH, mHCO3, lnacfHCO3, mCO3, lnacfCO3,
#            lnkHCO3)
#    else:
#        gH2CO3 = 0.0
#        gHCO3 = 0.0
#    if tots1[eles == 't_BOH3'] > 0:
#        gBOH3 = _GibbsBOH3(lnaw, lnacfBOH4, mBOH4, lnacfBOH3, mBOH3,
#            lnacfH, mH, lnkBOH3)
#    else:
#        gBOH3 = 0.0
    return lnacfs, lnaw, GComponents

allmxs = pz.matrix.assemble(allions, tempK, pres, prmlib=prmlib)
lnks = [lnkfunc(tempK, pres) for lnkfunc in lnkfuncs]

allmols, allions = getallmols(eqstate, tots, eles, eqindex, equilibria,
    eqXCO3, eqXF, eqtots, zbfixed, eqions, fixmols, fixions)

lnacfs, lnaw, GComponents = newGibbs(eqstate, tots, eles, eqindex, equilibria,
    eqXCO3, eqXF, eqtots, zbfixed, eqions, fixmols, fixions, allmxs, lnks,
    tots, ideal=False)

