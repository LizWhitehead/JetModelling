/* KSYNCH (JHC c. 2013) */

/* Generates profiles in kappa (proton/electron energy ratio), and corresponding B and energy densities, for Models I-IV of Croston & Hardcastle (2014) MNRAS 438, 4 */

/* INPUT is plain-text file: four columns - rmid (kpc) = distance along jet of the middle of the jet slice, radio frequency (in Hz), synchrotron emissivity (S(nu)*(1+z)^5/(theta^3 * D_L)) or (S(nu)*4*Pi*DL*DL*(1+z)/(vol in SI), and external pressure (Pa) */

/* OUTPUT is plain-text file: column are Jdist = jet distance (i.e. rmid); beqnprot = B for equipartition and k=0; pintnoprot = internal pressure for equipartition and no protons; krel = kappa for relativistic protons and pressure balance (Model III); beqrprot = B_eq for Model III; uer = U_e for Model III; ubr = U_B for Model III; ur = U_p for Model III; kth = kappa for thermal protons and pressure balance (Model IV); beqtprot = B_eq for Model IV; uet = U_e for Model IV; ubt = U_B for Model IV; uth = U_p for Model IV; bdome = B for lepton dominance (Model I), uedome = U_E for Model I; ubdome = U_B for Model I; bdomb = B for magnetic dominance (Model II); uedomb = U_E for Model II; ubdomb = U_B for Model II. All B in Tesla and all U in J/m^3. */

#include <Python.h>

/*#include <unistd.h>*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "integ.h"

#define PI 3.1415927
#define G 6.672e-11
#define MU 0.60
#define MH 1.67e-27
#define NMAX 1000
#define C 3.0e8
#define QELEC 1.602e-19
#define KP 0.0449
#define MELEC 9.11e-31
#define EPS_0 8.85e-12
#define MU_0 1.256e-6
#define PIND 2.1 /* electron energy injection index; can be varied */
#define EMIN 8e-13 /* electron energy lower cutoff ; can be varied */
#define EMAX 8e-8 /* electron energy higher cutoff ; can be varied */
#define LOGKMIN 0.0
#define LOGKMAX 4.0 /* maximum allowed kappa ; can be varied */
#define IMAX 1000
#define TOL 0.001
#define KB 1.38e-23
#define TENV 1.36e3 /* environmental temperature ; can be varied */
#define V0 1.5e8 /* inner velocity bound ; can be varied */
#define TOL2 0.01

void fit(float [], float [], int, float [], int, float *, float *, float *, float *, float *, float *);
void nrerror();
float *vector();
float **matrix();
float **submatrix();
float **convert_matrix();
float ***f3tensor();
double *dvector();
double **dmatrix();
int *ivector();
int **imatrix();
unsigned char *cvector();
unsigned long *lvector();
void free_vector();
void free_dvector();
void free_ivector();
void free_cvector();
void free_lvector();
void free_matrix();
void free_submatrix();
void free_convert_matrix();
void free_dmatrix();
void free_imatrix();
void free_f3tensor();

void rstring(FILE *f,char *s) {

  int c;
  c=-1;
  while (c) {
      c=fgetc(f);
      if ((c==EOF) || (c==10))
          {
           *s=0;
            c=0; }
      else *s=c;
    s++;
      }
}

int signum(double x)

{
  
  if(x>0) {
    
    return 1;
    
  }  else if(x<0) {
    
    return -1;
    
  }  else {
    
    return 0;
  }
}

void main (int argc, char *argv[]) {

  double pext[NMAX],rfreq[NMAX],remiss[NMAX];
  double kval[NMAX],krel[NMAX];
  double rpin[NMAX],pgrad[NMAX],vol[NMAX],rjet[NMAX];
  double xarea[NMAX],rxt[NMAX];
   double c1, c2, c3, c4, c5, c6, cinn,kmid, kfunc, ksol;
 static char *sepstr=" ,()";
 int i,j,k,nelems,npelems,iter,xtelems;
  char *s;
  char buffer[10240];
  FILE *inprof, *pprof, *pout, *xtab;
   double minfunc,maxfunc,lkmin,lkmax,lkminnl,lkmaxnl;
   double beqnoprot[NMAX], xrang;
   double uet[NMAX],uth[NMAX],ubt[NMAX],uer[NMAX],ur[NMAX],ubr[NMAX];
   double uenoprot[NMAX], ubnoprot[NMAX], pintnoprot[NMAX];
   double uedome[NMAX], ubdome[NMAX], uedomb[NMAX], ubdomb[NMAX];
   double bdome[NMAX], bdomb[NMAX];
   double beqtprot[NMAX], beqrprot[NMAX];
   double bmid,utest,eu,el,bmin,bmax,kin,cn,eint;

  /* input file: four columns - rmid (kpc), radio frequency (in Hz), synchrotron emissivity (S(nu)*(1+z)^5/(theta^3 * D_L)) or (S(nu)*4*Pi*DL*DL*(1+z)/(vol in SI), volume (m^3) and external pressures (Pa) */

  inprof=fopen(argv[1],"r");
  if(!inprof) {
    fprintf(stderr,"Failed to read input file - quitting!\n");
    exit(0);
  }

  i=0;
  while(!feof(inprof)) {
    rstring(inprof,buffer);
    /*    printf("Reading row %i\n",i); */
   if(buffer[0] && buffer[0]!='#'){
      s=strtok(buffer,sepstr);
      sscanf(s,"%lg",&rjet[i]);
      s=strtok(NULL,sepstr);
      sscanf(s,"%lg",&rfreq[i]); 
      s=strtok(NULL,sepstr);
      sscanf(s,"%lg",&remiss[i]); 
      /* s=strtok(NULL,sepstr);
      sscanf(s,"%lg",&vol[i]); */
      s=strtok(NULL,sepstr); 
      sscanf(s,"%lg",&pext[i]);
    i++;
    }
  }
  /*  printf("Read in profile\n"); */
  nelems = i;

  /* Iterate through jet slices, solving for k_rel and k_th for the two cases of B in equipartiton with the electrons, and B in equipartition with all particles  */

  cinn=pow(MELEC,3.0)*pow(C,4.0)/QELEC;
  c1= KP*(pow(QELEC,3.0)/(EPS_0*C*MELEC))*pow(cinn,-(PIND-1)/2.0);

  for(j=0;j<nelems;j++) {

    lkmin=LOGKMIN;
    lkmax=LOGKMAX;

    c3 = remiss[j]*pow(rfreq[j],(PIND - 1)/2.0)*(pow(EMAX,2 - PIND) - pow(EMIN,2 - PIND))/(c1*(2 - PIND));

    c4 = pow(2.0*MU_0*c3,4.0/(PIND+5))/(2.0*MU_0);

    c2 = 3*pext[j];

    c5 = pow(2.0*MU_0*c3,-(PIND+1)/(PIND+5)); 

    eint = (pow(EMAX,2 - PIND) - pow(EMIN,2 - PIND))/(2 - PIND);

    cn = remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*eint/c1;

    /*    printf("c1=%g,c2=%g,c3=%g,c4=%g,c5=%g,c6=%g\n",c1,c2,c3,c4,c5,c6); */


   
    iter=0;
    while(iter<IMAX) { 
      /* stepping through in log space to find k_thermal */
      lkminnl=pow(10.0,lkmin);
      lkmaxnl=pow(10.0,lkmax);
      kmid = pow(10.0,(lkmax+lkmin)/2.0);

      kfunc = c2 - pow(2.0*MU_0,-(PIND+1)/(PIND+5))*pow(c3,4.0/(PIND+5))*(pow(1+kmid,4.0/(PIND+5))+ (2*kmid+1)*pow(1.0 + kmid,-(PIND+1)/(PIND+5)));
      minfunc = c2 - pow(2.0*MU_0,-(PIND+1)/(PIND+5))*pow(c3,4.0/(PIND+5))*(pow(1+lkminnl,4.0/(PIND+5))+ (2*lkminnl+1)*pow(1.0 + lkminnl,-(PIND+1)/(PIND+5)));
      maxfunc = c2 - pow(2.0*MU_0,-(PIND+1)/(PIND+5))*pow(c3,4.0/(PIND+5))*(pow(1+lkmaxnl,4.0/(PIND+5))+ (2*lkmaxnl+1)*pow(1.0 + lkmaxnl,-(PIND+1)/(PIND+5)));

      /*      printf("lkminnl=%g,lkmaxnl=%g,kmid=%g, kfunc=%g, minfunc=%g\n",lkminnl,lkmaxnl,kmid,kfunc,minfunc); */
      if((kfunc == 0)||((lkmax-lkmin)/2.0<TOL)) {
	ksol = kmid;
	/*	printf("Found solution: k = %f; iter=%i\n",kmid,iter); */
	break;
      } else {
	if(signum(kfunc)==signum(minfunc)) {
	  lkmin=log10(kmid);
	} else {
	  lkmax=log10(kmid);
	}
	
      }
      
      iter++;
    }
    if(iter==IMAX) {
      /* no soluation found */
      kval[j] = -999;
    } else {
      kval[j] = ksol; // kappa_thermal

    }
    krel[j] = pow(3.0*pext[j]/2.0,(PIND+5)/4.0)*pow(2.0*MU_0,(PIND+1)/4.0)/cn - 1.0; // kappa_rel
    
    /* B and energy calculations for both thermal and relc proton cases (Models III and IV) */
    
    beqtprot[j] = pow(2*MU_0*kval[j]*remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*(pow(EMAX,2-PIND)-pow(EMIN,2-PIND))/(c1*(2-PIND)),2/(PIND+5));
    beqrprot[j] = pow(2*MU_0*krel[j]*remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*(pow(EMAX,2-PIND)-pow(EMIN,2-PIND))/(c1*(2-PIND)),2/(PIND+5));
    uet[j] = remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*pow(beqtprot[j],-(PIND+1)/2.0)*(pow(EMAX,2-PIND)-pow(EMIN,2-PIND))/(c1*(2-PIND));
    uer[j] = remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*pow(beqrprot[j],-(PIND+1)/2.0)*(pow(EMAX,2-PIND)-pow(EMIN,2-PIND))/(c1*(2-PIND));
    uth[j]=(kval[j]-1)*uet[j];
    ur[j] = (krel[j]-1)*uer[j];
    ubt[j]=beqtprot[j]*beqtprot[j]/(2.0*MU_0);
    ubr[j]=beqrprot[j]*beqrprot[j]/(2.0*MU_0);    

    /* output values for equipartition and no protons */

    beqnoprot[j] = pow(2*MU_0*remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*(pow(EMAX,2-PIND)-pow(EMIN,2-PIND))/(c1*(2-PIND)),2/(PIND+5));
    uenoprot[j] = remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*pow(beqnoprot[j],-(PIND+1)/2.0)*(pow(EMAX,2-PIND)-pow(EMIN,2-PIND))/(c1*(2-PIND));
    ubnoprot[j] = beqnoprot[j]*beqnoprot[j]/(2.0*MU_0);
    pintnoprot[j] = (1/3.0)*uenoprot[j] + (1/3.0)*ubnoprot[j];

    /* pressure matching: electron dominated and B dominated cases (Models I and II) - just uses bisector method */

    bmin = 1.0e-13; /* range for e-dom case */
    bmax = 5.0e-9;
   
    eu =  (1/3.0)*remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*pow(bmin,-(PIND+1)/2.0)*(pow(EMAX,2-PIND) - pow(EMIN,2-PIND))/(c1*(2.0-PIND)) + (1/3.0)*bmin*bmin/(2.0*MU_0);
    el =  (1/3.0)*remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*pow(bmax,-(PIND+1)/2.0)*(pow(EMAX,2-PIND) - pow(EMIN,2-PIND))/(c1*(2.0-PIND)) + (1/3.0)*bmax*bmax/(2.0*MU_0); 
    utest = 1.0e99;

    k=0;
    while(fabs((utest-3.0*pext[j])/(3.0*pext[j]))>TOL2) {

      bmid = exp((log(bmin)+log(bmax))/2.0); 
      /* calculate energy density at midpoint */
      
      uedome[j] = remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*pow(bmid,-(PIND+1)/2.0)*(pow(EMAX,2-PIND) - pow(EMIN,2-PIND))/(c1*(2.0-PIND));
      ubdome[j] = bmid*bmid/(2.0*MU_0);
      utest = uedome[j]+ubdome[j];
      
      if(utest<(3*pext[j])) {
	/* pressure too low - need to lower B field */
	bmax = bmid;
      } else {
	/* pressure too high - increasing B contribution */
	bmin = bmid;
      }
      /*    printf("Iteration %i, bmid = %g, utest = %g, 3*pext = %g\n",k,bmid,utest,3.0*pext[j]); */
      k++;
    }

    bdome[j] = bmid;

    bmin = 1.0e-9; /* range for B-dom case */
    bmax = 1.0e-6;
   
    utest = 1.0e99;

    k=0;
    while(fabs((utest-3.0*pext[j])/(3.0*pext[j]))>TOL2) {
      bmid = exp((log(bmin)+log(bmax))/2.0);  

      /* calculate energy density at midpoint */
      
      uedomb[j] = remiss[j]*pow(rfreq[j],(PIND-1)/2.0)*pow(bmid,-(PIND+1)/2.0)*(pow(EMAX,2-PIND) - pow(EMIN,2-PIND))/(c1*(2.0-PIND));
      ubdomb[j] = bmid*bmid/(2.0*MU_0);
      utest = uedomb[j]+ubdomb[j];

      if(utest<(3.0*pext[j])) {
	/* pressure too low - need to increase B field */
	bmin = bmid;
      } else {
	bmax = bmid;
      }
 
      k++;
    }

    bdomb[j] = bmid;

  }

  /* write to output file */
  
  printf("Jdist beqnprot pintnoprot krel beqrprot uer ubr ur kth beqtprot uet ubt uth bdome uedome ubdome bdomb uedomb ubdomb\n");

  for(j=0;j<nelems;j++) {

    printf("%.3f %.3g %.3g %.3f %.3g %.3g %.3g %.3g %.3f %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g\n",rjet[j],beqnoprot[j],pintnoprot[j],krel[j],beqrprot[j],uer[j],ubr[j],ur[j],kval[j],beqtprot[j],uet[j],ubt[j],uth[j],bdome[j],uedome[j],ubdome[j],bdomb[j],uedomb[j],ubdomb[j],pext[j]);
  }

}

