from autograd import numpy as np
import pandas as pd
import pytzer as pz
from scipy import optimize
import pickle, time

cf = pz.cdicts.CRP94

T,tots,ions,idf = pz.io.getIons('CRP94 solver.csv')

# Solve for pH - CRP94 system
def minifun(mH,tots,ions,T,cf):
    
    # Calculate [H+] and ionic speciation
#    mH = np.vstack(-np.log10(pH))
    mH = np.vstack(mH)
    mHSO4 = 2*tots - mH
    mSO4  = mH - tots
    
    # Create molality & ions arrays
    mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
    ions = np.array(['H','HSO4','SO4'])
    
    # Calculate activity coefficients
    ln_acfs = pz.model.ln_acfs(mols,ions,T,cf)
    gH    = np.exp(ln_acfs[:,0])
    gHSO4 = np.exp(ln_acfs[:,1])
    gSO4  = np.exp(ln_acfs[:,2])

    # Set up DG equation
#    DG = np.log(gH*mH.ravel() * gSO4*mSO4.ravel() / (gHSO4*mHSO4.ravel())) \
#        - np.log(cf.K['HSO4'](T)[0])
    
    DG = cf.getKeq(T, mH=mH,gH=gH, mHSO4=mHSO4,gHSO4=gHSO4, 
                   mSO4=mSO4,gSO4=gSO4)
    
    return DG

go = time.time()

def get_mH(tots,ions,T,cf):
    
#    fgo = time.time()
    
    mH = np.vstack(np.full_like(T,np.nan))
    for i in range(len(mH)):
        
        iT = np.array([T[i]])
        itots = np.array([tots[i,:]])
        
        mH[i] = optimize.least_squares(lambda mH: minifun(mH,itots,ions,iT,cf),
                                       1.5*itots[0],
                                       bounds=(itots[0],2*itots[0]),
                                       method='trf',
                                       xtol=1e-12)['x']
    
#        print(time.time()-fgo)
    
    return mH

mH = get_mH(tots,ions,T,cf)
mH2 = pz.data.dis_sim_H2SO4(tots,T,cf)

print(time.time()-go)

# Get solution - all looking good in test case
mHSO4 = 2*tots - mH
mSO4  = mH - tots

alpha = mSO4 / tots

mols = np.concatenate((mH,mHSO4,mSO4), axis=1)
ions = np.array(['H', 'HSO4', 'SO4'])

osm = pz.model.osm(mols,ions,T,cf)
osmST = osm * (mH + mHSO4 + mSO4).ravel() / (3 * tots.ravel())

acfs = pz.model.acfs(mols,ions,T,cf)
acfPM = np.cbrt(acfs[:,0]**2 * acfs[:,2] * mH.ravel()**2 * mSO4.ravel() \
    / (4 * tots.ravel()**3))

data = pd.DataFrame({'T':T, 'TSO4':tots.ravel(), 'alpha':alpha.ravel(),
                     'osm':osm, 'osmST':osmST,
                     'mH':mH.ravel(), 'mHSO4':mHSO4.ravel(),
                     'mSO4':mSO4.ravel(), 'gTSO4':acfPM,
                     'gH':acfs[:,0], 'gHSO4':acfs[:,1], 'gSO4':acfs[:,2]})

# Simulate and pickle a dataset for fit testing
s_TSO4 = np.vstack(np.arange(0.01,np.sqrt(6),0.01, dtype='float64')**2)
s_T = np.full_like(s_TSO4.ravel(),298.15, dtype='float64')

#s_mH = get_mH(s_TSO4,ions,s_T,cf)
s_mH = pz.data.dis_sim_H2SO4(s_TSO4,s_T,cf)

#with open('pickles/sulfit.pkl','wb') as f:
#    pickle.dump((s_T,s_TSO4,s_mH),f)