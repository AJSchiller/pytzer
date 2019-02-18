from pandas import read_excel
from numpy import append, array, concatenate, float_, logical_not, mean, \
                  ones_like, pi, size, sqrt, std, unique, vstack

#from numpy import abs as np_abs
#from numpy import max as np_max
from numpy import sum as np_sum

import pytzer as pz
#from scipy.special import factorial

from copy import deepcopy

# Choose electrolyte to examine
testele = 'NaOH'

cf = deepcopy(pz.cfdicts.MarChemSpec)

#cf.ions = concatenate([cf.ions, array(['Li','I','Rb','Cs'])])
#cf.add_zeros(cf.ions)

#cf.bC['Na-Cl'] = pz.coeffs.bC_Na_Cl_A92ii
#cf.bC['K-Cl' ] = pz.coeffs.bC_K_Cl_A99

#cf.bC['Li-Cl'] = pz.coeffs.bC_Li_Cl_HM83
#cf.bC['Cs-Cl'] = pz.coeffs.bC_Cs_Cl_HM83
    
e2i = {'NaOH': [array(['Na','OH']), float_([1,1])],
                  
       'K2CO3': [array(['K','CO3']), float_([2,1])],
                
       'Zn(NO3)2': [array(['Znjj','NO3']), float_([1,2])],
       
       'H2SO4' : [array(['H'   ,'SO4']), float_([2,1])],
       'Li2SO4': [array(['Li'  ,'SO4']), float_([2,1])],
       'Na2SO4': [array(['Na'  ,'SO4']), float_([2,1])],
       'MgSO4' : [array(['Mg'  ,'SO4']), float_([1,1])],
       'K2SO4' : [array(['K'   ,'SO4']), float_([2,1])],
       'CuSO4' : [array(['Cujj','SO4']), float_([1,1])],
       
       'Na2S2O3': [array(['Na','S2O3']), float_([2,1])],
       
       'Mg(NO3)2' : [array(['Mg'  ,'NO3']), float_([1,2])],
       'Sr(NO3)2' : [array(['Sr'  ,'NO3']), float_([1,2])],
       'Ba(NO3)2' : [array(['Ba'  ,'NO3']), float_([1,2])],
       'Co(NO3)2' : [array(['Cojj','NO3']), float_([1,2])],
       'Cu(NO3)2' : [array(['Cujj','NO3']), float_([1,2])],
       'Cd(NO3)2' : [array(['Cdjj','NO3']), float_([1,2])],
       'UO2(NO3)2': [array(['UO2' ,'NO3']), float_([1,2])],
       
       'NaF': [array(['Na','F']), float_([1,1])],
       'KF' : [array(['K' ,'F']), float_([1,1])],
       
       'LiCl' : [array(['Li'  ,'Cl']), float_([1,1])],
       'NaCl' : [array(['Na'  ,'Cl']), float_([1,1])],
       'MgCl2': [array(['Mg'  ,'Cl']), float_([1,2])],
       'KCl'  : [array(['K'   ,'Cl']), float_([1,1])],
       'CaCl2': [array(['Ca'  ,'Cl']), float_([1,2])],
       'CuCl2': [array(['Cujj','Cl']), float_([1,2])],
       'RbCl' : [array(['Rb'  ,'Cl']), float_([1,1])],
       'CsCl' : [array(['Cs'  ,'Cl']), float_([1,1])],
       'LaCl3': [array(['La'  ,'Cl']), float_([1,3])],
       
       'Mg(ClO4)2': [array(['Mg'  ,'ClO4']), float_([1,2])],
       'Zn(ClO4)2': [array(['Znjj','ClO4']), float_([1,2])],
       
       'LiI': [array(['Li','I']), float_([1,1])],
       
       'tris'       : [array(['tris'       ]), float_([1  ])],
       '(trisH)2SO4': [array(['trisH','SO4']), float_([2,1])],
       'trisHCl'    : [array(['trisH','Cl' ]), float_([1,1])],
       
       'glycerol': [array(['glycerol']), float_([1])],
       'sucrose' : [array(['sucrose' ]), float_([1])],
       'urea'    : [array(['urea'    ]), float_([1])]}

filename = '../pytzer-support/datasets/isonew.xlsx'

isonew = read_excel(filename, skiprows=2)

isobase = isonew[['src','temp']]

isodict = {}

for r in range(len(isonew.index)):

    irow = isonew.iloc[r,6:]
    
    isodict[r] = {}
    
    for i, icell in enumerate(irow):
    
        if type(icell) is str:
            
            isodict[r][icell] = {}
            irc = isodict[r][icell]
            
            celes = icell.split('-')
            
            irc['tots'] = irow[i-len(celes):i].values.astype('float64')
                        
            irc_ions = concatenate([e2i[cele][0] for cele in celes])
                            
            irc_mols = concatenate([e2i[cele][1] * irc['tots'][c] \
                for c, cele in enumerate(celes)])
            
            irc['ions'] = unique(irc_ions)
            
            irc['mols'] = array([[np_sum(irc_mols[irc_ions == ion]) \
                for ion in irc['ions']]])
            
            irc['T'] = float_([[isonew.temp[r]]])
    
#%% Get all data for a particular electrolyte

# Extract subset of isodict
testdict = {irow: isodict[irow]             \
            for irow    in isodict.keys()      \
             if testele in isodict[irow].keys()}

# Fill out cfdict with zeros if necessary
testions = unique(concatenate([testdict[irow][icell]['ions'] \
           for irow  in testdict.keys()                      \
           for icell in testdict[irow].keys()]))
cf.ions = unique(append(cf.ions,testions))
cf.add_zeros(cf.ions)

# Calculate activities
for irow in testdict.keys():
    
    for icell in testdict[irow].keys():
    
        trc = testdict[irow][icell]
        
        # Calculate ionic strengths
        trc['zs']   = pz.props.charges(trc['ions'])[0]
        trc['Istr'] = pz.model.Istr(trc['mols'],trc['zs'])
        
        # Calculate osmotic coefficients
        trc['osm'] = pz.model.osm(trc['mols'],trc['ions'],trc['T'],cf,
                                  Izero = trc['Istr'] == 0)
        
        # Convert into water activities
        trc['aw'] = pz.model.osm2aw(trc['mols'],trc['osm'])
        

# Calculate aw difference        
for irow in testdict.keys():
    
    for icell in testdict[irow].keys():
        
        if icell != testele:
            
            trc = testdict[irow][icell]
            
            trc['del_aw'] = trc['aw'] - testdict[irow][testele]['aw']
  
    
#%% Get arrays for plotting
t_dictrow = vstack([irow for irow  in testdict.keys()
                         for icell in testdict[irow].keys()
                          if icell != testele])

t_elemix = array([icell for irow  in testdict.keys()
                        for icell in testdict[irow].keys()
                         if icell != testele])
    
t_testtot = vstack([testdict[irow][testele]['tots']
                    for irow in testdict.keys()
                    for icell in testdict[irow].keys()
                     if icell != testele])
    
t_T = vstack([testdict[irow][testele]['T']
              for irow in testdict.keys()
              for icell in testdict[irow].keys()
               if icell != testele])
    
t_delaw = vstack([testdict[irow][icell]['del_aw']
                  for irow  in testdict.keys()
                  for icell in testdict[irow].keys()
                   if icell != testele])

from scipy.io import savemat
savemat('testfiles/isonew_' + testele + '.mat',
        {'dictrow': t_dictrow,
         'elemix' : t_elemix ,
         'testtot': t_testtot,
         'T'      : t_T      ,
         'delaw'  : t_delaw  })
    
#%%
#            irc['osm'] = pz.model.osm(irc['mols'],irc['ions'],irc['T'],cf)
#            
#            irc['aw'] = pz.model.osm2aw(irc['mols'],irc['osm'])
#
## Get all water activities            
#all_aw = [concatenate(
#    [isodict[irow][icell]['aw'] for icell in isodict[irow].keys()]).ravel() \
#     for irow in isodict.keys()]
#
#aw_mean = array([mean(aw) for aw in all_aw])
#aw_sd   = array([std (aw) for aw in all_aw])
#ncups   = array([size(aw) for aw in all_aw])
#
#def std_unbias(stds,nobs):
#
#    def keven(k2):
#        
#        k = k2/2
#        
#        return sqrt(2 / (pi * (2*k - 1))) * 2**(2*k - 2) \
#            * factorial(k - 1)**2 / factorial(2*k - 2)
#    
#    def kodd(k2p1):
#        
#        k = (k2p1 - 1)/2
#        
#        return sqrt(pi / k) * factorial(2*k - 1) \
#            / (2**(2*k - 1) * factorial(k - 1)**2)
#            
#    c4 = ones_like(nobs, dtype='float64')
#    
#    Leven = nobs % 2 == 0
#    Lodd  = logical_not(Leven)
#    
#    c4[Leven] = keven(nobs[Leven])
#    c4[Lodd ] = kodd (nobs[Lodd ])
#    
#    return stds / c4
#            
#aw_sdu = std_unbias(aw_sd,ncups)
#
##%%
#from matplotlib import pyplot as plt
#
#plt.scatter(range(len(aw_sdu)),aw_sdu)
##plt.scatter(aw_mean,aw_sdu)
#
#sdu_max = np_abs(np_max(aw_sdu)) * 1.1
#
#plt.ylim([0,sdu_max])
#