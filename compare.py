import numpy as np
import pytzer as pz

# Import Clegg's bC and ao values
clegg_filename = 'testfiles/threeway/FastPitz05.Rs1'

clegg_bC_ions = np.genfromtxt(clegg_filename, skip_header=35, 
                              skip_footer=86+638,
                              usecols=range(5,7), dtype='U')

clegg_bC_ions = np.vstack(['-'.join(ions) for ions in clegg_bC_ions])

clegg_bC = np.genfromtxt(clegg_filename, skip_header=35, skip_footer=86+638,
                         usecols=range(5))

#clegg_ao = np.genfromtxt(clegg_filename, skip_header=67, skip_footer=57+7,
#                         usecols=range(4))

# Get coefficients from pytzer.cfdicts.WM13
cf = pz.cfdicts.MarChemSpec05

T = np.float_(278.15)

WM13_bC_ions = np.vstack([ions.upper() for ions in cf.bC.keys()])

WM13_bC = np.vstack([[bC for bC in cf.bC[key](T)[:5]] for key in cf.bC.keys()])

# Compare the two

print('Coefficient dictionary: {} [pytzer-v{}]\n'.format(cf.name,
      pz.meta.version))

for cnum in range(5):

    cnames = ['beta-0','beta-1','beta-2','C-0','C-1']
    
    print('====================================================')
    print('{:^52}'.format('Coefficient: ' + cnames[cnum]))
    print('----------------------------------------------------')
    print('{:>10} {:^12} {:^12} {:^12}'.format('Ions','pytzer',
                                               'FastPitz','fp-pz'))
    
    for ix,ions in enumerate(WM13_bC_ions):
        
        L = (clegg_bC_ions == ions).ravel()
        test = clegg_bC[L]
        
        strfmt = '{:>10}' + 3*'{:>13.5e}' + '  {}'
            
        cdiff = test[0][cnum] - WM13_bC[ix][cnum]
        
        note = ' '
        if np.abs(cdiff) > np.abs(1e-6 * WM13_bC[ix][cnum]):
            note = '*'
        
        print(strfmt.format(ions[0],WM13_bC[ix][cnum],test[0][cnum],cdiff,note))
    
    print('====================================================')
    print('')