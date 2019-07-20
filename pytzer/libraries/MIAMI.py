# Pytzer: Pitzer model for chemical activities in aqueous solutions.
# Copyright (C) 2019  Matthew Paul Humphreys  (GNU GPLv3)
from .. import parameters as prm
from .. import debyehueckel, unsymmetrical
name = 'MIAMI'
dh = {'Aosm': debyehueckel.Aosm_M88}
jfunc = unsymmetrical.Harvie
bC = {}
theta = {}
psi = {}
# Table A1
bC['Na-Cl'] = prm.bC_Na_Cl_M88
bC['K-Cl'] = prm.bC_K_Cl_GM89
bC['K-SO4'] = prm.bC_K_SO4_GM89
bC['Ca-Cl'] = prm.bC_Ca_Cl_GM89
bC['Ca-SO4'] = prm.bC_Ca_SO4_M88
bC['Ca-SO3'] = prm.bC_Ca_SO3_MP98
bC['Sr-SO4'] = prm.bC_Sr_SO4_MP98
# Table A2
bC['Mg-Cl'] = prm.bC_Mg_Cl_PP87i
bC['Mg-SO4'] = prm.bC_Mg_SO4_PP86ii
# Table A3
bC['Na-HSO4'] = prm.bC_Na_HSO4_MP98
bC['Na-HCO3'] = prm.bC_Na_HCO3_PP82
bC['Na-SO4'] = prm.bC_Na_SO4_HPR93
bC['Na-CO3'] = prm.bC_Na_CO3_PP82
bC['Na-BOH4'] = prm.bC_Na_BOH4_SRRJ87
bC['Na-HS'] = prm.bC_Na_HS_HPM88
bC['Na-SCN'] = prm.bC_Na_SCN_SP78
bC['Na-SO3'] = prm.bC_Na_SO3_MHJZ89
bC['Na-HSO3'] = prm.bC_Na_HSO3_MHJZ89
# Table A4
bC['K-HCO3'] = prm.bC_K_HCO3_RGWW83
bC['K-CO3'] = prm.bC_K_CO3_SRG87
bC['K-BOH4'] = prm.bC_K_BOH4_SRRJ87
bC['K-HS'] = prm.bC_K_HS_HPM88
bC['K-H2PO4'] = prm.bC_K_H2PO4_SP78
bC['K-SCN'] = prm.bC_K_SCN_SP78
# Table A5
#bC['Mg-Br'] = prm.bC_Mg_Br_SP78
bC['Mg-BOH4'] = prm.bC_Mg_BOH4_SRM87
#bC['Mg-ClO4'] = prm.bC_Mg_ClO4_SP78
#bC['Ca-Br'] = prm.bC_Ca_Br_SP78
bC['Ca-BOH4'] = prm.bC_Ca_BOH4_SRM87
#bC['Ca-ClO4'] = prm.bC_Ca_ClO4_SP78
# Table A6
bC['Sr-Br'] = prm.bC_Sr_Br_SP78
bC['Sr-Cl'] = prm.bC_Sr_Cl_SP78 # not in table but in text §4.6
#bC['Sr-NO3' ] = prm.bC_Sr_NO3_SP78
#bC['Sr-ClO4'] = prm.bC_Sr_ClO4_SP78
#bC['Sr-HSO3'] = prm.bC_Sr_HSO3_SP78
bC['Sr-BOH4'] = prm.bC_Sr_BOH4_MP98
# Table A7
bC['Na-I'] = prm.bC_Na_I_MP98
bC['Na-Br'] = prm.bC_Na_Br_MP98
bC['Na-F'] = prm.bC_Na_F_MP98
bC['K-Br'] = prm.bC_K_Br_MP98
bC['K-F'] = prm.bC_K_F_MP98
bC['K-OH'] = prm.bC_K_OH_MP98
bC['K-I'] = prm.bC_K_I_MP98
bC['Na-ClO3'] = prm.bC_Na_ClO3_MP98
bC['K-ClO3'] = prm.bC_K_ClO3_MP98
bC['Na-ClO4'] = prm.bC_Na_ClO4_MP98
bC['Na-BrO3'] = prm.bC_Na_BrO3_MP98
bC['K-BrO3'] = prm.bC_K_BrO3_MP98
bC['Na-NO3'] = prm.bC_Na_NO3_MP98
bC['K-NO3'] = prm.bC_K_NO3_MP98
bC['Mg-NO3'] = prm.bC_Mg_NO3_MP98
bC['Ca-NO3'] = prm.bC_Ca_NO3_MP98
bC['H-Br'] = prm.bC_H_Br_MP98
bC['Sr-Cl'] = prm.bC_Sr_Cl_MP98
bC['NH4-Cl'] = prm.bC_NH4_Cl_MP98
bC['NH4-Br'] = prm.bC_NH4_Br_MP98
bC['NH4-F'] = prm.bC_NH4_F_MP98
# Table A8
bC.update({iset: lambda T: prm.bC_PM73(T, iset)
    for iset in ['Sr-I', 'Na-NO2', 'Na-H2PO4', 'Na-HPO4', 'Na-PO4',
        'K-NO2', 'K-HPO4', 'K-PO4', 'Mg-I', 'Ca-I', 'SO4-NH4']})
#bC['Na-H2AsO4' ] = prm.bC_Na_H2AsO4_PM73
#bC['K-HAsO4'   ] = prm.bC_K_HAsO4_PM73
#bC['Na-HAsO4'  ] = prm.bC_Na_HAsO4_PM73
#bC['Na-AsO4'   ] = prm.bC_Na_AsO4_PM73
#bC['Na-acetate'] = prm.bC_Na_acetate_PM73
bC['K-HSO4' ] = prm.bC_K_HSO4_HMW84
#bC['K-AsO4'    ] = prm.bC_K_AsO4_PM73
#bC['K-acetate' ] = prm.bC_K_acetate_PM73
bC['Mg-HS'  ] = prm.bC_Mg_HS_HPM88
bC['Ca-HSO4'] = prm.bC_Ca_HSO4_HMW84
bC['Ca-HCO3'] = prm.bC_Ca_HCO3_HMW84
bC['Ca-HS'  ] = prm.bC_Ca_HS_HPM88
bC['Ca-OH'  ] = prm.bC_Ca_OH_HMW84
bC['MgOH-Cl'] = prm.bC_MgOH_Cl_HMW84
# Table A9
bC['H-Cl' ] = prm.bC_H_Cl_CMR93
#bC['H-SO4'] = prm.bC_H_SO4_Pierrot
# Table A10
theta['Cl-CO3' ] = prm.theta_Cl_CO3_PP82
theta['Cl-HCO3'] = prm.theta_Cl_HCO3_PP82
# Table A11
theta['Cl-SO3'] = prm.theta_Cl_SO3_MHJZ89
psi['Na-Cl-SO3'] = prm.psi_Na_Cl_SO3_MHJZ89