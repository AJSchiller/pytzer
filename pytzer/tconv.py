from autograd import numpy as np
from autograd import elementwise_grad as egrad
from . import model
from .constants import Mw, R

##### FREEZING POINT DEPRESSION ###############################################

# Convert freezing point depression to water activity
def fpd2aw(fpd):

    # Equation coefficients from S.L. Clegg (pers. comm., 2018)
    lg10aw = \
        - np.float_(4.209099e-03) * fpd    \
        - np.float_(2.151997e-06) * fpd**2 \
        + np.float_(3.233395e-08) * fpd**3 \
        + np.float_(3.445628e-10) * fpd**4 \
        + np.float_(1.758286e-12) * fpd**5 \
        + np.float_(7.649700e-15) * fpd**6 \
        + np.float_(3.117651e-17) * fpd**7 \
        + np.float_(1.228438e-19) * fpd**8 \
        + np.float_(4.745221e-22) * fpd**9
    
    return np.exp(lg10aw * np.log(10))

# Convert freezing point depression to osmotic coefficient
def fpd2osm(mols,fpd):
    return model.aw2osm(mols,fpd2aw(fpd))

##### TEMPERATURE CONVERSION ##################################################

# --- Temperature subfunctions ------------------------------------------------
    
def y(T0,T1):
    return (T1 - T0) / (R * T0 * T1)

def z(T0,T1):
    return T1 * y(T0,T1) - np.log(T1 / T0) / R

def O(T0,T1):
    return T1 * (z(T0,T1) + (T0 - T1)*y(T0,T1) / 2)

# --- Heat capacity derivatives -----------------------------------------------

# wrt. molality
dCpapp_dm = egrad(model.Cpapp)

def J1(tot,nC,nA,ions,T,cf): # HO58 Ch. 8 Eq. (8-4-9)
    return -Mw * tot**2 * dCpapp_dm(tot,nC,nA,ions,T,cf)

def J2(tot,nC,nA,ions,T,cf): # HO58 Ch. 8 Eq. (8-4-7)
    return tot * dCpapp_dm(tot,nC,nA,ions,T,cf)

# wrt. temperature
G1 = egrad(J1, argnum=4)
G2 = egrad(J2, argnum=4)

# --- Enthalpy derivatives ----------------------------------------------------
    
# wrt. molality
dLapp_dm = egrad(model.Lapp)

def L1(tot,nC,nA,ions,T,cf): # HO58 Ch. 8 Eq. (8-4-9)
    return -Mw * tot**2 * dLapp_dm(tot,nC,nA,ions,T,cf)

def L2(tot,nC,nA,ions,T,cf): # HO58 Ch. 8 Eq. (8-4-7)
    return    model.Lapp(tot,nC,nA,ions,T,cf) \
        + tot * dLapp_dm(tot,nC,nA,ions,T,cf)

# --- Execute temperature conversion ------------------------------------------

# Osmotic coefficient
def osm2osm(tot,nC,nA,ions,T0,T1,TR,cf,osm_T0):
    
    tot = np.vstack(tot)
    T0  = np.vstack(T0)
    T1  = np.vstack(T1)
    TR  = np.vstack(TR)
    
    lnAW_T0 = -osm_T0 * tot * (nC + nA) * Mw
    
    lnAW_T1 = lnAW_T0 - y(T0,T1) * L1(tot,nC,nA,ions,TR,cf) \
                      + z(T0,T1) * J1(tot,nC,nA,ions,TR,cf) \
                      - O(T0,T1) * G1(tot,nC,nA,ions,TR,cf)

    return -lnAW_T1 / (tot * (nC + nA) * Mw)
    
# Solute mean activity coefficient
def acf2acf(tot,nC,nA,ions,T0,T1,TR,cf,acf_T0):
    
    tot = np.vstack(tot)
    T0  = np.vstack(T0)
    T1  = np.vstack(T1)
    TR  = np.vstack(TR)
    
    ln_acf_T0 = np.log(acf_T0)
    
    ln_acf_T1 = ln_acf_T0 + (- y(T0,T1) * L2(tot,nC,nA,ions,TR,cf) \
                          + z(T0,T1) * J2(tot,nC,nA,ions,TR,cf) \
                          - O(T0,T1) * G2(tot,nC,nA,ions,TR,cf)) / (nC + nA)
    
    return np.exp(ln_acf_T1)
