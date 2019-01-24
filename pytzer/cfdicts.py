# pytzer: the Pitzer model for chemical speciation
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)

from . import coeffs, jfuncs, props
from .meta import version
from numpy import array
from copy import deepcopy

#==============================================================================
#===================================== Define CoefficientDictionary class =====

class CoefficientDictionary:

    # Initialise
    def __init__(self):
        self.name  = ''
        self.dh    = {} # Aosm
        self.bC    = {} # c-a
        self.theta = {} # c-c' and a-a'
        self.jfunc = [] # unsymmetrical mixing
        self.psi   = {} # c-c'-a and c-a-a'
        self.lambd = {} # n-c and n-a
        self.eta   = {} # n-c-a
        self.mu    = {} # n-n-n

    # Populate with zero-functions
    def add_nones(self,ions):

        # Get lists of cations and anions
        _,cations,anions,neutrals = props.charges(ions)

        # Sort lists into alphabetical order
        cations.sort()
        anions.sort()

        # Populate cfdict with zero functions where no function exists

        # betas and Cs
        for cation in cations:
            for anion in anions:

                istr = '-'.join((cation,anion))
                if istr not in self.bC.keys():
                    self.bC[istr] = coeffs.bC_none

        # c-c'-a thetas and psis
        for C0, cation0 in enumerate(cations):
            for cation1 in cations[C0+1:]:

                istr = '-'.join((cation0,cation1))
                if istr not in self.theta.keys():
                    self.theta[istr] = coeffs.theta_none

                for anion in anions:

                    istr = '-'.join((cation0,cation1,anion))
                    if istr not in self.psi.keys():
                        self.psi[istr] = coeffs.psi_none

        # c-a-a' thetas and psis
        for A0, anion0 in enumerate(anions):
            for anion1 in anions[A0+1:]:

                istr = '-'.join((anion0,anion1))
                if istr not in self.theta.keys():
                    self.theta[istr] = coeffs.theta_none

                for cation in cations:

                    istr = '-'.join((cation,anion0,anion1))
                    if istr not in self.psi.keys():
                        self.psi[istr] = coeffs.psi_none

        # Neutral interactions
        for neutral in neutrals:

            # n-c lambdas
            for cation in cations:
                inc = '-'.join((neutral,cation))
                if inc not in self.lambd.keys():
                    self.lambd[inc] = coeffs.lambd_none

                # n-c-a etas
                for anion in anions:
                    inca = '-'.join((neutral,cation,anion))
                    if inca not in self.eta.keys():
                        self.eta[inca] = coeffs.eta_none

            # n-a lambdas
            for anion in anions:
                ina = '-'.join((neutral,anion))
                if ina not in self.lambd.keys():
                    self.lambd[ina] = coeffs.lambd_none

            # n-n-n mus
            innn = '-'.join((neutral,neutral,neutral))
            if innn not in self.mu.keys():
                self.mu[innn] = coeffs.mu_none

    # Print all coefficient values at a given temperature
    def print_coeffs(self,T,filename):

        f = open(filename,'w')

        f.write('Coefficient dictionary: {} [pytzer-v{}]\n\n'.format( \
                self.name,version))

        # Debye-Hueckel slope
        f.write('Debye-Hueckel limiting slope\n')
        f.write('============================\n')

        eval_Aosm = self.dh['Aosm'](T)[0]

        src = self.dh['Aosm'].__name__.split('_')[-1]

        f.write('{:^12}  {:15}\n'.format('Aosm','source'))
        f.write('{:>12.9f}  {:15}\n'.format(eval_Aosm,src))

        # Write cation-anion coefficients (betas and Cs)
        f.write('\n')
        f.write('c-a pairs (betas and Cs)\n')
        f.write('========================\n')
        
        bChead = 2*'{:7}' + 5*'{:^13}'    + 3*'{:>6}'    + '  {:15}\n'
        bCvals = 2*'{:7}' + 5*'{:>13.5e}' + 3*'{:>6.1f}' + '  {:15}\n'
        f.write(bChead.format('cat','ani', 'b0','b1','b2','C0','C1',
                              'al1','al2','omg', 'source'))

        for bC in self.bC.keys():

            cation,anion = bC.split('-')
            b0,b1,b2,C0,C1, alph1,alph2,omega, _ = self.bC[bC](T)

            src = self.bC[bC].__name__.split('_')[-1]

            f.write(bCvals.format(cation,anion, b0,b1,b2,C0,C1,
                                  alph1,alph2,omega, src))

        # Write same charge ion-ion coefficients (thetas)
        f.write('\n')
        f.write('c-c\' and a-a\' pairs (thetas)\n')
        f.write('============================\n')

        thetaHead = 2*'{:7}' + '{:^13}'    + '  {:15}\n'
        thetaVals = 2*'{:7}' + '{:>13.5e}' + '  {:15}\n'
        
        f.write(thetaHead.format('ion1','ion2','theta','source'))

        for theta in self.theta.keys():

            ion0,ion1 = theta.split('-')
            eval_theta = self.theta[theta](T)[0]

            src = self.theta[theta].__name__.split('_')[-1]

            f.write(thetaVals.format(ion0,ion1, eval_theta, src))

        # Write ion triplet coefficients (psis)
        f.write('\n')
        f.write('c-c\'-a and c-a-a\' triplets (psis)\n')
        f.write('=================================\n')

        psiHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        psiVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'

        f.write(psiHead.format('ion1','ion2','ion3','psi','source'))

        for psi in self.psi.keys():

            ion0,ion1,ion2 = psi.split('-')
            eval_psi = self.psi[psi](T)[0]

            src = self.psi[psi].__name__.split('_')[-1]

            f.write(psiVals.format(ion0,ion1,ion2,eval_psi,src))

        # Write neutral-ion coefficients (lambdas)
        f.write('\n')
        f.write('n-c and n-a pairs (lambdas)\n')
        f.write('===========================\n')

        lambdHead = 2*'{:7}' + '{:^13}'    + '  {:15}\n'
        lambdVals = 2*'{:7}' + '{:>13.5e}' + '  {:15}\n'
        
        f.write(lambdHead.format('neut','ion','lambda','source'))

        for lambd in self.lambd.keys():

            neut,ion = lambd.split('-')
            eval_lambd = self.lambd[lambd](T)[0]

            src = self.lambd[lambd].__name__.split('_')[-1]

            f.write(lambdVals.format(neut,ion, eval_lambd, src))

        # Write neutral-cation-anion triplet coefficients (etas)
        f.write('\n')
        f.write('n-c-a triplets (etas)\n')
        f.write('=====================\n')

        etaHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        etaVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'

        f.write(etaHead.format('neut','cat','ani','eta','source'))

        for eta in self.eta.keys():

            neut,cat,ani = eta.split('-')
            eval_eta = self.eta[eta](T)[0]

            src = self.eta[eta].__name__.split('_')[-1]

            f.write(etaVals.format(neut,cat,ani,eval_eta,src))

        # Write neutral-neutral-neutral triplet coefficients (mus)
        f.write('\n')
        f.write('n-n-n triplets (mus)\n')
        f.write('====================\n')

        muHead = 3*'{:7}' + '{:^12}'    + '  {:15}\n'
        muVals = 3*'{:7}' + '{:>12.5e}' + '  {:15}\n'

        f.write(muHead.format('neut1','neut2','neut3','mu','source'))

        for mu in self.mu.keys():

            neut1,neut2,neut3 = mu.split('-')
            eval_mu = self.mu[mu](T)[0]

            src = self.mu[mu].__name__.split('_')[-1]

            f.write(muVals.format(neut1,neut2,neut3,eval_mu,src))



#==============================================================================
#=============================== Define specific coefficient dictionaries =====

#------------------------------------------------------------ Møller 1988 -----

# Møller (1988). Geochim. Cosmochim. Acta 52, 821-837,
#  doi:10.1016/0016-7037(88)90354-7
#
# System: Na-Ca-Cl-SO4

M88 = CoefficientDictionary()
M88.name = 'M88'

# Debye-Hueckel limiting slope
M88.dh['Aosm'] = coeffs.Aosm_M88

# Cation-anion interactions (betas and Cs)
M88.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_M88
M88.bC['Ca-SO4'] = coeffs.bC_Ca_SO4_M88
M88.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88
M88.bC['Na-SO4'] = coeffs.bC_Na_SO4_M88

# Cation-cation and anion-anion interactions (theta)
# c-c'
M88.theta['Ca-Na' ] = coeffs.theta_Ca_Na_M88
# a-a'
M88.theta['Cl-SO4'] = coeffs.theta_Cl_SO4_M88

# Unsymmetrical mixing functions
M88.jfunc = jfuncs.Harvie

# Triplet interactions (psi)
# c-c'-a
M88.psi['Ca-Na-Cl' ] = coeffs.psi_Ca_Na_Cl_M88
M88.psi['Ca-Na-SO4'] = coeffs.psi_Ca_Na_SO4_M88
# c-a-a'
M88.psi['Ca-Cl-SO4'] = coeffs.psi_Ca_Cl_SO4_M88
M88.psi['Na-Cl-SO4'] = coeffs.psi_Na_Cl_SO4_M88


#------------------------------------------------ Greenberg & Møller 1989 -----

# Greenberg & Møller (1988). Geochim. Cosmochim. Acta 53, 2503-2518,
#  doi:10.1016/0016-7037(89)90124-5
#
# System: Na-K-Ca-Cl-SO4

GM89 = CoefficientDictionary()
GM89.name = 'GM89'

# Debye-Hueckel limiting slope
GM89.dh['Aosm'] = coeffs.Aosm_M88

# Cation-anion interactions (betas and Cs)
GM89.bC['Ca-Cl' ] = coeffs.bC_Ca_Cl_GM89
GM89.bC['Ca-SO4'] = coeffs.bC_Ca_SO4_M88
GM89.bC['K-Cl'  ] = coeffs.bC_K_Cl_GM89
GM89.bC['K-SO4' ] = coeffs.bC_K_SO4_GM89
GM89.bC['Na-Cl' ] = coeffs.bC_Na_Cl_M88
GM89.bC['Na-SO4'] = coeffs.bC_Na_SO4_M88

# Cation-cation and anion-anion interactions (theta)
# c-c'
GM89.theta['Ca-K'  ] = coeffs.theta_Ca_K_GM89
GM89.theta['Ca-Na' ] = coeffs.theta_Ca_Na_M88
GM89.theta['K-Na'  ] = coeffs.theta_K_Na_GM89
# a-a'
GM89.theta['Cl-SO4'] = coeffs.theta_Cl_SO4_M88

# Unsymmetrical mixing terms
GM89.jfunc = jfuncs.Harvie

# Triplet interactions (psi)
# c-c'-a
GM89.psi['Ca-K-Cl'  ] = coeffs.psi_Ca_K_Cl_GM89
GM89.psi['Ca-K-SO4' ] = coeffs.psi_Ca_K_SO4_GM89
GM89.psi['Ca-Na-Cl' ] = coeffs.psi_Ca_Na_Cl_M88
GM89.psi['Ca-Na-SO4'] = coeffs.psi_Ca_Na_SO4_M88
GM89.psi['K-Na-Cl'  ] = coeffs.psi_K_Na_Cl_GM89
GM89.psi['K-Na-SO4' ] = coeffs.psi_K_Na_SO4_GM89
# c-a-a'
GM89.psi['Ca-Cl-SO4'] = coeffs.psi_Ca_Cl_SO4_M88
GM89.psi['K-Cl-SO4' ] = coeffs.psi_K_Cl_SO4_GM89
GM89.psi['Na-Cl-SO4'] = coeffs.psi_Na_Cl_SO4_M88


#------------------------------------------------------ Clegg et al. 1994 -----

# Clegg et al. (1994). J. Chem. Soc., Faraday Trans. 90, 1875-1894,
#  doi:10.1039/FT9949001875
#
# System: H-HSO4-SO4

CRP94 = CoefficientDictionary()
CRP94.name = 'CRP94'

# Debye-Hueckel limiting slope
CRP94.dh['Aosm'] = coeffs.Aosm_CRP94

# Cation-anion interactions (betas and Cs)
CRP94.bC['H-HSO4'] = coeffs.bC_H_HSO4_CRP94
CRP94.bC['H-SO4' ] = coeffs.bC_H_SO4_CRP94

# Cation-cation and anion-anion interactions (theta)
# a-a'
CRP94.theta['HSO4-SO4'] = coeffs.theta_HSO4_SO4_CRP94

# Unsymmetrical mixing terms
CRP94.jfunc = jfuncs.P75_eq47

# Triplet interactions (psi)
# c-a-a'
CRP94.psi['H-HSO4-SO4'] = coeffs.psi_H_HSO4_SO4_CRP94


#-------------------------------------------------- Waters & Millero 2013 -----

# Waters and Millero (2013). Mar. Chem. 149, 8-22,
#  doi:10.1016/j.marchem.2012.11.003

WM13 = CoefficientDictionary()
WM13.name = 'WM13'

# Debye-Hueckel limiting slope and unsymmetrical mixing
WM13.dh['Aosm'] = coeffs.Aosm_M88
WM13.jfunc = jfuncs.P75_eq47

# Table A1: Na salts
WM13.bC['Na-Cl'  ] = coeffs.bC_Na_Cl_M88
WM13.bC['Na-SO4' ] = coeffs.bC_Na_SO4_HM86
WM13.bC['Na-HSO4'] = coeffs.bC_Na_HSO4_HPR93viaWM13
WM13.bC['Na-OH'  ] = coeffs.bC_Na_OH_PP87i

# Table A2: Mg salts
WM13.bC['Mg-Cl'  ] = coeffs.bC_Mg_Cl_dLP83
WM13.bC['Mg-SO4' ] = coeffs.bC_Mg_SO4_PP86ii
WM13.bC['Mg-HSO4'] = coeffs.bC_Mg_HSO4_RC99

# Table A3: Ca salts
WM13.bC['Ca-Cl'  ] = coeffs.bC_Ca_Cl_GM89
WM13.bC['Ca-SO4' ] = coeffs.bC_Ca_SO4_P91
WM13.bC['Ca-HSO4'] = coeffs.bC_Ca_HSO4_P91
WM13.bC['Ca-OH'  ] = coeffs.bC_Ca_OH_HMW84

# Table A4: K salts
WM13.bC['K-Cl'  ] = coeffs.bC_K_Cl_GM89
WM13.bC['K-SO4' ] = coeffs.bC_K_SO4_HM86
WM13.bC['K-HSO4'] = coeffs.bC_K_HSO4_P91
WM13.bC['K-OH'  ] = coeffs.bC_K_OH_HMW84

# Table A5: H+ interactions
WM13.bC['H-Cl'  ] = coeffs.bC_H_Cl_CMR93
WM13.bC['H-SO4' ] = coeffs.bC_H_SO4_CRP94
WM13.bC['H-HSO4'] = coeffs.bC_H_HSO4_CRP94

# Table A6: MgOH+ interactions
WM13.bC['MgOH-Cl'] = coeffs.bC_MgOH_Cl_HMW84

# Table A7: cation-cation interactions
WM13.theta['H-Na' ] = coeffs.theta_H_Na_CMR93
WM13.theta['H-Mg' ] = coeffs.theta_H_Mg_RGB80
WM13.theta['Ca-H' ] = coeffs.theta_Ca_H_RGO82 # WM13 citation error
WM13.theta['H-K'  ] = coeffs.theta_H_K_CMR93
WM13.theta['Mg-Na'] = coeffs.theta_Mg_Na_HMW84
WM13.theta['Ca-Na'] = coeffs.theta_Ca_Na_HMW84
WM13.theta['K-Na' ] = coeffs.theta_K_Na_HMW84
WM13.theta['Ca-Mg'] = coeffs.theta_Ca_Mg_HMW84
WM13.theta['K-Mg' ] = coeffs.theta_K_Mg_HMW84
WM13.theta['Ca-K' ] = coeffs.theta_Ca_K_HMW84

# Table A7: anion-anion interactions
WM13.theta['Cl-SO4'  ] = coeffs.theta_Cl_SO4_HMW84
WM13.theta['Cl-HSO4' ] = coeffs.theta_Cl_HSO4_HMW84
WM13.theta['Cl-OH'   ] = coeffs.theta_Cl_OH_HMW84
WM13.theta['HSO4-SO4'] = coeffs.theta_HSO4_SO4_WM13
WM13.theta['OH-SO4'  ] = coeffs.theta_OH_SO4_HMW84

# Table A8: c-a-a' triplets
WM13.psi['H-Cl-SO4' ] = coeffs.psi_H_Cl_SO4_WM13 # agrees with HMW84
WM13.psi['Na-Cl-SO4'] = coeffs.psi_Na_Cl_SO4_HMW84
WM13.psi['Mg-Cl-SO4'] = coeffs.psi_Mg_Cl_SO4_HMW84
WM13.psi['Ca-Cl-SO4'] = coeffs.psi_Ca_Cl_SO4_HMW84
WM13.psi['K-Cl-SO4' ] = coeffs.psi_K_Cl_SO4_HMW84

WM13.psi['H-Cl-HSO4' ] = coeffs.psi_H_Cl_HSO4_HMW84
WM13.psi['Na-Cl-HSO4'] = coeffs.psi_Na_Cl_HSO4_HMW84
WM13.psi['Mg-Cl-HSO4'] = coeffs.psi_Mg_Cl_HSO4_HMW84
WM13.psi['Ca-Cl-HSO4'] = coeffs.psi_Ca_Cl_HSO4_HMW84
WM13.psi['K-Cl-HSO4' ] = coeffs.psi_K_Cl_HSO4_HMW84

WM13.psi['H-Cl-OH' ] = coeffs.psi_H_Cl_OH_WM13 # agrees with HMW84
WM13.psi['Na-Cl-OH'] = coeffs.psi_Na_Cl_OH_HMW84
WM13.psi['Mg-Cl-OH'] = coeffs.psi_Mg_Cl_OH_WM13 # agrees with HMW84
WM13.psi['Ca-Cl-OH'] = coeffs.psi_Ca_Cl_OH_HMW84
WM13.psi['K-Cl-OH' ] = coeffs.psi_K_Cl_OH_HMW84

WM13.psi['H-HSO4-SO4' ] = coeffs.psi_H_HSO4_SO4_HMW84
WM13.psi['Na-HSO4-SO4'] = coeffs.psi_Na_HSO4_SO4_HMW84
WM13.psi['Mg-HSO4-SO4'] = coeffs.psi_Mg_HSO4_SO4_RC99
WM13.psi['Ca-HSO4-SO4'] = coeffs.psi_Ca_HSO4_SO4_WM13 # agrees with HMW84
WM13.psi['K-HSO4-SO4' ] = coeffs.psi_K_HSO4_SO4_HMW84

WM13.psi['H-OH-SO4' ] = coeffs.psi_H_OH_SO4_WM13 # agrees with HMW84
WM13.psi['Na-OH-SO4'] = coeffs.psi_Na_OH_SO4_HMW84
WM13.psi['Mg-OH-SO4'] = coeffs.psi_Mg_OH_SO4_WM13 # agrees with HMW84
WM13.psi['Ca-OH-SO4'] = coeffs.psi_Ca_OH_SO4_WM13 # agrees with HMW84
WM13.psi['K-OH-SO4' ] = coeffs.psi_K_OH_SO4_HMW84

# Table A9: c-c'-a triplets
WM13.psi['H-Na-Cl'  ] = coeffs.psi_H_Na_Cl_HMW84
WM13.psi['H-Na-SO4' ] = coeffs.psi_H_Na_SO4_WM13 # agrees with HMW84
WM13.psi['H-Na-HSO4'] = coeffs.psi_H_Na_Cl_HMW84

WM13.psi['H-Mg-Cl'] = coeffs.psi_H_Mg_Cl_HMW84
WM13.psi['H-Mg-SO4'] = coeffs.psi_H_Mg_SO4_RC99
WM13.psi['H-Mg-HSO4'] = coeffs.psi_H_Mg_HSO4_RC99

WM13.psi['Ca-H-Cl'  ] = coeffs.psi_Ca_H_Cl_HMW84
WM13.psi['Ca-H-SO4' ] = coeffs.psi_Ca_H_SO4_WM13 # agrees with HMW84
WM13.psi['Ca-H-HSO4'] = coeffs.psi_Ca_H_HSO4_WM13 # agrees with HMW84

WM13.psi['H-K-Cl'  ] = coeffs.psi_H_K_Cl_HMW84
WM13.psi['H-K-SO4' ] = coeffs.psi_H_K_SO4_HMW84
WM13.psi['H-K-HSO4'] = coeffs.psi_H_K_HSO4_HMW84

WM13.psi['Mg-Na-Cl'  ] = coeffs.psi_Mg_Na_Cl_HMW84
WM13.psi['Mg-Na-SO4' ] = coeffs.psi_Mg_Na_SO4_HMW84
WM13.psi['Mg-Na-HSO4'] = coeffs.psi_Mg_Na_HSO4_WM13 # agrees with HMW84

WM13.psi['Ca-Na-Cl'  ] = coeffs.psi_Ca_Na_Cl_HMW84
WM13.psi['Ca-Na-SO4' ] = coeffs.psi_Ca_Na_SO4_HMW84
WM13.psi['Ca-Na-HSO4'] = coeffs.psi_Ca_Na_HSO4_WM13 # agrees with HMW84

WM13.psi['K-Na-Cl'  ] = coeffs.psi_K_Na_Cl_HMW84
WM13.psi['K-Na-SO4' ] = coeffs.psi_K_Na_SO4_HMW84
WM13.psi['K-Na-HSO4'] = coeffs.psi_K_Na_HSO4_WM13 # agrees with HMW84

WM13.psi['Ca-Mg-Cl'  ] = coeffs.psi_Ca_Mg_Cl_HMW84
WM13.psi['Ca-Mg-SO4' ] = coeffs.psi_Ca_Mg_SO4_HMW84
WM13.psi['Ca-Mg-HSO4'] = coeffs.psi_Ca_Mg_HSO4_WM13 # agrees with HMW84

WM13.psi['K-Mg-Cl'  ] = coeffs.psi_K_Mg_Cl_HMW84
WM13.psi['K-Mg-SO4' ] = coeffs.psi_K_Mg_SO4_HMW84
WM13.psi['K-Mg-HSO4'] = coeffs.psi_K_Mg_HSO4_WM13 # agrees with HMW84

WM13.psi['Ca-K-Cl'  ] = coeffs.psi_Ca_K_Cl_HMW84
WM13.psi['Ca-K-SO4' ] = coeffs.psi_Ca_K_SO4_WM13 # agrees with HMW84
WM13.psi['Ca-K-HSO4'] = coeffs.psi_Ca_K_HSO4_WM13 # agrees with HMW84


#------------------------------------------------------------ MarChemSpec -----

# Begin with WM13
MarChemSpec = deepcopy(WM13)
MarChemSpec.name = 'MarChemSpec'

# Add coefficients from GT17 Supp. Info. Table S6 (simultaneous optimisation)
MarChemSpec.bC['Na-Cl'    ] = coeffs.bC_Na_Cl_GT17simopt
MarChemSpec.bC['trisH-SO4'] = coeffs.bC_trisH_SO4_GT17simopt
MarChemSpec.bC['trisH-Cl' ] = coeffs.bC_trisH_Cl_GT17simopt

MarChemSpec.theta['H-trisH'] = coeffs.theta_H_trisH_GT17simopt

MarChemSpec.psi['H-trisH-Cl'] = coeffs.psi_H_trisH_Cl_GT17simopt

MarChemSpec.lambd['tris-trisH'] = coeffs.lambd_tris_trisH_GT17simopt
MarChemSpec.lambd['tris-Na'   ] = coeffs.lambd_tris_Na_GT17simopt
MarChemSpec.lambd['tris-K'    ] = coeffs.lambd_tris_K_GT17simopt
MarChemSpec.lambd['tris-Mg'   ] = coeffs.lambd_tris_Mg_GT17simopt
MarChemSpec.lambd['tris-Ca'   ] = coeffs.lambd_tris_Ca_GT17simopt

MarChemSpec.add_nones(array(['H','Na','Mg','Ca','K','MgOH','trisH','Cl','SO4',
                             'HSO4','OH','tris']))

#==============================================================================
