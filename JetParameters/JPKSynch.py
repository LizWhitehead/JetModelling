import JetModelling_SourceSetup as JSS
from builtins import object
from math import fabs, log10, log, exp

# KSYNCH (JHC c. 2013)

# Generates profiles in kappa (proton/electron energy ratio), and corresponding B and energy densities, for Models I-IV of Croston & Hardcastle (2014) MNRAS 438, 4

# INPUT for a jet slice: radio frequency (in Hz), synchrotron emissivity (S(nu)*(1+z)^5/(theta^3 * D_L)) or (S(nu)*4*Pi*DL*DL*(1+z)/(vol in SI), and external pressure (Pa)

# OUTPUT for a jet slice: 
# beqnprot = B for equipartition and k=0
# pintnoprot = internal pressure for equipartition and no protons
# krel = kappa for relativistic protons and pressure balance (Model III)
# beqrprot = B_eq for Model III
# uer = U_e for Model III
# ubr = U_B for Model III 
# ur = U_p for Model III
# kth = kappa for thermal protons and pressure balance (Model IV)
# beqtprot = B_eq for Model IV
# uet = U_e for Model IV
# ubt = U_B for Model IV
# uth = U_p for Model IV
# bdome = B for lepton dominance (Model I)
# uedome = U_E for Model I
# ubdome = U_B for Model I
# bdomb = B for magnetic dominance (Model II)
# uedomb = U_E for Model II
# ubdomb = U_B for Model II
# [All B in Tesla and all U in J/m^3]

PI = 3.1415927
G = 6.672e-11
MU = 0.60
MH = 1.67e-27
C = 3.0e8
QELEC = 1.602e-19
KP = 0.0449
MELEC = 9.11e-31
EPS_0 = 8.85e-12
MU_0 = 1.256e-6
PIND = (2 * JSS.spectral_index) + 1     # electron energy injection index
EMIN = JSS.emin                         # electron energy lower cutoff
EMAX = JSS.emax                         # electron energy higher cutoff
LOGKMIN = 0.0
LOGKMAX = JSS.logkmax                   # maximum allowed kappa
IMAX = 1000
TOL = 0.001
KB = 1.38e-23
TOL2 = 0.01

class KSynch(object):
    def __init__(self):
        None

    def signum(self, x):
        if (x > 0):
            return 1
        elif (x < 0):
            return -1
        else:
            return 0

    def ksynch_calculate (self, rfreq, remiss, pext):

        # Input for a jet slice: radio frequency (in Hz), synchrotron emissivity (S(nu)*(1+z)^5/(theta^3 * D_L)) or (S(nu)*4*Pi*DL*DL*(1+z)/(vol in SI), volume (m^3) and external pressures (Pa)
        # Solve for k_rel and k_th for the two cases of B in equipartiton with the electrons, and B in equipartition with all particles

        cinn = pow(MELEC, 3.0) * pow(C, 4.0) / QELEC
        c1= KP * ( pow(QELEC, 3.0) / (EPS_0 * C * MELEC) ) * pow(cinn, -(PIND-1)/2.0)

        lkmin = LOGKMIN
        lkmax = LOGKMAX

        c3 = remiss * pow(rfreq, (PIND - 1)/2.0) * (pow(EMAX, 2 - PIND) - pow(EMIN, 2 - PIND)) / (c1 * (2 - PIND))
        c4 = pow(2.0 * MU_0 * c3, 4.0 / (PIND+5)) / (2.0 * MU_0)
        c2 = 3 * pext
        c5 = pow(2.0 * MU_0 * c3, -(PIND+1) / (PIND + 5))
        eint = (pow(EMAX, 2 - PIND) - pow(EMIN, 2 - PIND)) / (2 - PIND)
        cn = remiss * pow(rfreq, (PIND-1)/2.0) * eint / c1

        # print("c1=%g,c2=%g,c3=%g,c4=%g,c5=%g,c6=%g\n",c1,c2,c3,c4,c5)
   
        icnt = 0
        while (icnt < IMAX):
            # stepping through in log space to find k_thermal
            lkminnl = pow(10.0, lkmin)
            lkmaxnl = pow(10.0, lkmax)
            kmid = pow(10.0, (lkmax + lkmin)/2.0)

            kfunc = c2 - pow(2.0 * MU_0, -(PIND+1) / (PIND+5)) * pow(c3, 4.0 / (PIND+5)) * (pow(1.0 + kmid, 4.0 / (PIND+5)) + (2.0 * kmid + 1.0) * pow(1.0 + kmid, -(PIND+1) / (PIND+5)))
            minfunc = c2 - pow(2.0 * MU_0, -(PIND+1) / (PIND+5)) * pow(c3, 4.0 / (PIND+5)) * (pow(1.0 + lkminnl, 4.0 / (PIND+5)) + (2.0 * lkminnl + 1.0) * pow(1.0 + lkminnl, -(PIND+1) / (PIND+5)))
            maxfunc = c2 - pow(2.0 * MU_0, -(PIND+1) / (PIND+5)) * pow(c3, 4.0 / (PIND+5)) * (pow(1.0 + lkmaxnl, 4.0 / (PIND+5)) + (2.0 * lkmaxnl + 1.0) * pow(1.0 + lkmaxnl, -(PIND+1) / (PIND+5)))

            # print("lkminnl=%g,lkmaxnl=%g,kmid=%g, kfunc=%g, minfunc=%g\n",lkminnl,lkmaxnl,kmid,kfunc,minfunc)
            if (kfunc == 0) or ((lkmax - lkmin) / 2.0 < TOL):
                ksol = kmid
                # print("Found solution: k = %f; icnt=%i\n",kmid,icnt)
                break
            elif (self.signum(kfunc) == self.signum(minfunc)):
                lkmin = log10(kmid)
            else:
                lkmax = log10(kmid)

            icnt += 1

        if(icnt == IMAX):
            # no soluation found
            self.kval = -999
        else:
            self.kval = ksol # kappa_thermal

        self.krel = pow(3.0 * pext/2.0, (PIND+5)/4.0) * pow(2.0 * MU_0, (PIND+1)/4.0) / cn - 1.0 # kappa_rel
    
        # B and energy calculations for both thermal and relc proton cases (Models III and IV)
    
        self.beqtprot = pow(2 * MU_0 * self.kval * remiss * pow(rfreq, (PIND-1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2-PIND)), 2 / (PIND+5))
        self.beqrprot = pow(2 * MU_0 * self.krel * remiss * pow(rfreq, (PIND-1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2-PIND)), 2 / (PIND+5))
        self.uet = remiss * pow(rfreq, (PIND-1)/2.0) * pow(self.beqtprot, -(PIND+1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2-PIND))
        self.uer = remiss * pow(rfreq, (PIND-1)/2.0) * pow(self.beqrprot, -(PIND+1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2-PIND))
        self.uth = (self.kval - 1) * self.uet
        self.ur = (self.krel - 1) * self.uer
        self.ubt = self.beqtprot * self.beqtprot / (2.0 * MU_0)
        self.ubr = self.beqrprot * self.beqrprot / (2.0 * MU_0)   

        # output values for equipartition and no protons

        self.beqnoprot = pow(2 * MU_0 * remiss * pow(rfreq, (PIND-1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2-PIND)), 2 / (PIND+5))
        uenoprot = remiss * pow(rfreq, (PIND-1)/2.0) * pow(self.beqnoprot, -(PIND+1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2-PIND))
        ubnoprot = self.beqnoprot * self.beqnoprot / (2.0 * MU_0)
        self.pintnoprot = (1/3.0) * uenoprot + (1/3.0) * ubnoprot

        # pressure matching: electron dominated and B dominated cases (Models I and II) - just uses bisector method

        bmin = 1.0e-13 # range for e-dom case
        bmax = 5.0e-9
   
        eu =  (1/3.0) * remiss * pow(rfreq, (PIND-1)/2.0) * pow(bmin, -(PIND+1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2.0 - PIND)) + (1/3.0) * bmin * bmin / (2.0 * MU_0)
        el =  (1/3.0) * remiss * pow(rfreq, (PIND-1)/2.0) * pow(bmax, -(PIND+1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2.0 - PIND)) + (1/3.0) * bmax * bmax / (2.0 * MU_0)
        utest = 1.0e99

        k = 0
        while (fabs((utest - 3.0 * pext) / (3.0 * pext)) > TOL2):
            bmid = exp( (log(bmin) + log(bmax)) / 2.0 )

            # calculate energy density at midpoint
      
            self.uedome = remiss * pow(rfreq, (PIND-1)/2.0) * pow(bmid, -(PIND+1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2.0-PIND))
            self.ubdome = bmid * bmid / (2.0 * MU_0)
            utest = self.uedome + self.ubdome
      
            if(utest < (3 * pext)):
                # pressure too low - need to lower B field
                bmax = bmid
            else:
                # pressure too high - increasing B contribution
                bmin = bmid

            # print("Iteration %i, bmid = %g, utest = %g, 3*pext = %g\n", k, bmid, utest, 3.0*pext)
            k += 1

        self.bdome = bmid

        bmin = 1.0e-9 # range for B-dom case
        bmax = 1.0e-6
   
        utest = 1.0e99

        k=0
        while (fabs((utest - 3.0 * pext) / (3.0 * pext)) > TOL2):
            bmid = exp( (log(bmin) + log(bmax)) / 2.0 )

            # calculate energy density at midpoint
      
            self.uedomb = remiss * pow(rfreq, (PIND-1)/2.0) * pow(bmid, -(PIND+1)/2.0) * (pow(EMAX, 2-PIND) - pow(EMIN, 2-PIND)) / (c1 * (2.0-PIND))
            self.ubdomb = bmid * bmid / (2.0 * MU_0)
            utest = self.uedomb + self.ubdomb

            if(utest < (3.0 * pext)):
                # pressure too low - need to increase B field
                bmin = bmid
            else:
                bmax = bmid
 
            k += 1

        self.bdomb = bmid

        # print("%.3g %.3g %.3f %.3g %.3g %.3g %.3g %.3f %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g\n", \
        #       self.beqnoprot,self.pintnoprot,self.krel,self.beqrprot,self.uer,self.ubr,self.ur,self.kval,self.beqtprot, \
        #       self.uet,self.ubt,self.uth,self.bdome,self.uedome,self.ubdome,self.bdomb,self.uedomb,self.ubdomb)

        return None
