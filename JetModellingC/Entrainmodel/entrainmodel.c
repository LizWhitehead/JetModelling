/* ENTRAIN (JHC c. 2013 */

/* Determines velocity and entrainment rate profiles self-consistently from input jet thermal pressure profile, following method of Croston & Hardcastle (2014) MNRAS 438, 4 */

/* INPUT: a set of input profiles in bins of distance along the jet:
   inprof = thermal pressure lookup table
   xtab = jet cross-section lookup table
   pprof = external pressure gradient lookup table
   blook = U_B lookup table
   elook = U_E lookup table
 */


/* OUTPUT:
   entrain.out: plain text files with columns rval = distance along jet (kpc), area2 = cross-sectional area of slice (m^2), tnew = assumed internal temperature (K), rhotherm = thermal pressure (kg/m^3), v = jet speed (m/s), P_buoy, Psi_ent, Psi_l, E_kin = kinetic energy flux (J/s), energy flux of thermal particles (J/s), energy flux of relativistic leptons (J/s), magnetic energy flux, total energy flux (J/s), work done on external medium / time (J/s) */

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
#define NMAX 1200
#define C 3.0e8
#define QELEC 1.602e-19
#define PIND 2.1
#define KP 0.0449
#define MELEC 9.11e-31
#define EPS_0 8.85e-12
#define MU_0 1.256e-6
#define EMIN 8e-13
#define EMAX 8e-8
#define LOGRMIN -30.0
#define LOGRMAX -25.0
#define IMAX 1000
#define TOL 0.001
#define KB 1.38e-23
#define TENV 1.36e3
// #define TEMP 10000.0
#define V0 6.0e7 // use value at 12 kpc
#define RSTART 12
#define REND 120
#define BINSIZE 2
// #define BINSIZE 4

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

  double ptherm[NMAX], rb[NMAX],re[NMAX], ubin[NMAX], uein[NMAX], xarea[NMAX], rxt[NMAX], rjet[NMAX], rpin[NMAX],pgrad[NMAX];
  double *rhotherm, *area2, *rin, *rout, *rval, *area, *pext;
  double *press, *work, *pbuoy, *psient, *psil, *comp;
  double *v, *ekin, *eintt, *eintb, *einte, *etot, *tnew, *ue, *uth, *ub;
  double c1, c2, c3, c4, c5, cn, eint, cinn,rhomid, rfunc, rsol, temp, turbratio;
 static char *sepstr=" ,()";
 int i,j,k,nelems,npelems,iter,xtelems,telems,eelems,belems;
  char *s;
  char buffer[10240];
  FILE *inprof, *pprof, *pout, *xtab,*blook,*elook, *ene;
  double minfunc,maxfunc,lrmin,lrmax,lrminnl,lrmaxnl,eold;
  double beq, xrang, val,cwork,ttest,vmin,vmax;

    /* NB. input profiles all need to use jet distance not radial distance */

  /* Input for southern jet
  
   inprof=fopen("ptherms_lookup.tab","r");
   pout=fopen("entrain.out","w");
   xtab=fopen("xsecs_good.tab","r");
   pprof=fopen("dpressdrs_newsi.txt","r");
   blook=fopen("ubs_lookup.tab","r");
   elook=fopen("ues_lookup.tab","r");
   ene=fopen("energetics.out","w"); */

  /* Input for northern jet */
   inprof=fopen("pthermn_lookup.tab","r");
   xtab=fopen("xsecn_good.tab","r");
   pprof=fopen("dpressdr_newsi.txt","r");
   blook=fopen("ub_lookup.tab","r");
   elook=fopen("ue_lookup.tab","r");
   pout=fopen("entrain.out","w");
   ene=fopen("energetics.out","w"); 

   if(!inprof) {
     fprintf(stderr,"Failed to read input file - quitting!\n");
     exit(0);
   }
   if(!pprof) {
     fprintf(stderr,"Failed to read pressure gradient file - quitting\n");
     exit(0);
   }

 i=0;
  while(!feof(inprof)) {
    rstring(inprof,buffer);
   if(buffer[0] && buffer[0]!='#'){
      s=strtok(buffer,sepstr);
      sscanf(s,"%lg",&rjet[i]);
      s=strtok(NULL,sepstr);
      sscanf(s,"%lg",&ptherm[i]); 
      i++;
    }
  }
  printf("Read in ptherm lookup table.\n");
  telems = i;

 i=0;
  while(!feof(pprof)) {
    rstring(pprof,buffer);
    /*    printf("Reading row %i\n",i); */
   if(buffer[0] && buffer[0]!='#'){
      s=strtok(buffer,sepstr);
      sscanf(s,"%lg",&rpin[i]);
      s=strtok(NULL,sepstr);
      sscanf(s,"%lg",&pgrad[i]); 
      i++;
    }
  }
  printf("Read in pressure gradient profile\n");
  npelems = i;
  printf("Pressure gradient profile has %i bins\n",npelems);

  /* Now allocate related arrays */

  area=calloc(npelems,sizeof(double));

  i=0;
  while(!feof(elook)) {
    rstring(elook,buffer);
    /*    printf("Reading row %i\n",i); */
   if(buffer[0] && buffer[0]!='#'){
      s=strtok(buffer,sepstr);
      sscanf(s,"%lg",&re[i]);
      s=strtok(NULL,sepstr);
      sscanf(s,"%lg",&uein[i]); 
      i++;
    }
  }

  printf("Read in Ue lookup table.\n");
  eelems = i;

 i=0;
  while(!feof(blook)) {
    rstring(blook,buffer);
    /*    printf("Reading row %i\n",i); */
   if(buffer[0] && buffer[0]!='#'){
      s=strtok(buffer,sepstr);
      sscanf(s,"%lg",&rb[i]);
      s=strtok(NULL,sepstr);
      sscanf(s,"%lg",&ubin[i]); 
      i++;
    }
  }
  printf("Read in UB lookup tables\n");
  belems = i;

  i=0;
  while(!feof(xtab)) {
    rstring(xtab,buffer);
      if(buffer[0] && buffer[0]!='#'){
      s=strtok(buffer,sepstr);
      sscanf(s,"%lg",&rxt[i]);
      s=strtok(NULL,sepstr);
      sscanf(s,"%lg",&xarea[i]); 
      i++;
    }
  }
  printf("Read in jet area look-up table\n");
  xtelems = i;
  printf("Area table has %i entries\n",xtelems);

  /* Step through jet bins, determine rhotherm for given T, determine buoyancy term, get velocity and entrainment profile */

  /* Start at 12 kpc, and go to 140 kpc in 1kpc steps */

  nelems=(REND-RSTART)/BINSIZE;
 
  /* Allocate related arrays */
  rhotherm=calloc(nelems,sizeof(double));
  area2=calloc(nelems,sizeof(double));
  rin=calloc(nelems,sizeof(double));
  rout=calloc(nelems,sizeof(double));
  rval=calloc(nelems,sizeof(double));
  press=calloc(nelems,sizeof(double));
  work=calloc(nelems,sizeof(double));
  pbuoy=calloc(nelems,sizeof(double));
  psient=calloc(nelems,sizeof(double));
  psil=calloc(nelems,sizeof(double));
  comp=calloc(nelems,sizeof(double));
  v=calloc(nelems,sizeof(double));
  ekin=calloc(nelems,sizeof(double));
  eintt=calloc(nelems,sizeof(double));
  eintb=calloc(nelems,sizeof(double));
  einte=calloc(nelems,sizeof(double));
  etot=calloc(nelems,sizeof(double));
  tnew=calloc(nelems,sizeof(double));
  ue=calloc(nelems,sizeof(double));
  uth=calloc(nelems,sizeof(double));
  ub=calloc(nelems,sizeof(double));
  pext=calloc(nelems,sizeof(double));

 for(i=0;i<nelems;i++) {

    rin[i] = (i*BINSIZE+RSTART);
    rout[i] = (i+1)*BINSIZE+RSTART;
    rval[i] = (rout[i]+rin[i])/2.0;
    val = rval[i];  
    /* look up pressure at midpoint of this bin */
    
    xrang=1e99;
    for(k=0;k<telems;k++) {
      if(abs(val-rjet[k])<xrang) {
	press[i] = ptherm[k];
	xrang = abs(val-rjet[k]);
	
      }
      
    }  
    /* look up Ue and Ub for this bin */
    xrang=1e99;
    for(k=0;k<eelems;k++) {
      if(abs(val-re[k])<xrang) {
	ue[i] = uein[k];
	xrang = abs(val-re[k]);
	
      }
      
    }  
    xrang=1e99;
    for(k=0;k<belems;k++) {
      if(abs(val-rb[k])<xrang) {
	ub[i] = ubin[k];
	xrang = abs(val-rb[k]);
	
      }
      
    }  
  
    
    //  rhotherm[i] = press[i]*(MU*MH)/(temp*QELEC);
    //  ttest = temp*pow(rval[i]/rval[0],0.5);
    //   printf("bin %i, r = %g, temp = %g, press = %g, rho = %g\n",i,rval[i],ttest,press[i],rhotherm[i]);
  }
    /* generate profile in A(l) for bins corresponding to P profile */
    /* NB - now array elements correspond to value at inner boundary of bin */
    
  for(j=0;j<npelems;j++) {
    
    /* Now use look-up table of cross-sectional areas instead of analytic form. Table is in units of arcsec and arcsec^2 */
    
    /*   area[j] = PI*pow((0.203*(rpin[j]/(0.3438*3.09e19)) - 0.534)*0.3438*(3.09e19),2.0); old analytic formula */
    
    xrang=1e99;
    for(k=0;k<xtelems;k++) {
      if(abs(rpin[j]-rxt[k])<xrang) {
	area[j] = xarea[k];
	xrang = abs(rpin[j]-rxt[k]);
	
      }
      
    }
  }
  for(j=0;j<nelems;j++) {
      
      /*   area2[j] =  PI*pow((0.203*(rjet[j]/(0.3438*3.09e19)) - 0.534)*0.3438*(3.09e19),2.0); old analytic formula */
      
      xrang=1e99;
      for(k=0;k<xtelems;k++) {
	//	printf("k=%g,xarea=%g\n",k,xarea[k]);
	if(fabs(rval[j]-rxt[k])<xrang) {
	  area2[j] = xarea[k];
	  xrang = fabs(rval[j]-rxt[k]);
	  //  printf("k=%i, xrang = %g, rxt=%g, xarea=%g\n",k, xrang,rxt[k],xarea[k]);
	}
      }
           printf("rval = %g, Final area %i is %g\n",rval[j],j,area2[j]);
  }
  
  /* Now integrate buoyancy term over larger bins of jet profile */
  /* why not just use narrow bins, given we have emissivity model? */
  for(j=0;j<nelems;j++) {
    pbuoy[j]=0.0;
    for(k=0;k<npelems;k++) {
      if((rpin[k]>rval[j])&&(rpin[k]<rval[j+1])) {
	pbuoy[j]+=pgrad[k]*area[k]*(rpin[k+1]-rpin[k])*(3.09e19);
      }
    }
    
  }
  
  /* And generate velocity and entrainment rate profile */
  v[0] = V0;
  rhotherm[0] = press[0]*(MU*MH)/(temp*QELEC);
  ekin[0] = 0.5*(rhotherm[0])*pow(v[0],3.0)*area2[0];
  eintt[0] = (3.0/2.0)*press[0]*area2[0]*v[0];
  einte[0] = ue[0]*area2[0]*v[0];
  eintb[0] = ub[0]*area2[0]*v[0];
  etot[0] = ekin[0] + eintt[0] + einte[0] + eintb[0];
  psient[0]=0.0;
  psil[0]=0.0;
  pext[0] = press[0] + (1/3.0)*(ue[0]+ub[0]);
  
  printf("Boundary conditions: v=%g,  rho=%g, area=%g, etot=%g, ekin=%g, eth=%g, ee=%g, eb=%g\n",v[0],rhotherm[0],area2[0],etot[0],ekin[0],eintt[0],einte[0],eintb[0]);

  // Go through and determine T based on conserving Etot, then work out velocity in the next bin

  for(j=1;j<nelems;j++) {
   
    // Use bisector method to solve for rhotherm, conserving energy and momentum
  
    lrmin = LOGRMIN;
    lrmax = LOGRMAX;
    iter=0;
    while(iter<IMAX) {
      lrminnl=pow(10.0,lrmin);
      lrmaxnl=pow(10.0,lrmax);
      rhomid = pow(10.0,(lrmax+lrmin)/2.0);
      
      v[j] = sqrt((rhotherm[j-1]*pow(v[j-1],2.0)*area2[j-1] + pbuoy[j])/(rhomid*area2[j]));
      ekin[j] = 0.5*rhomid*pow(v[j],3.0)*area2[j];
      einte[j] = ue[j]*area2[j]*v[j];
      eintb[j] = ub[j]*area2[j]*v[j];
      eintt[j] = (3.0/2.0)*press[j]*area2[j]*v[j];
      pext[j] = press[j] + (1/3.0)*(ue[j]+ub[j]);
      work[j] = pext[j]*area2[j]*v[j] - pext[j-1]*area2[j-1]*v[j-1];
    
      rfunc = -etot[0] + ekin[j] + eintt[j] + einte[j] + eintb[j]+work[j];
      vmax = sqrt((rhotherm[j-1]*pow(v[j-1],2.0)*area2[j-1]+pbuoy[j])/(lrmaxnl*area2[j]));
      vmin = sqrt((rhotherm[j-1]*pow(v[j-1],2.0)*area2[j-1]+pbuoy[j])/(lrminnl*area2[j]));
      maxfunc = -etot[0] + 0.5*lrmaxnl*pow(vmax,3.0)*area2[j] + (3.0/2.0)*press[j]*area2[j]*vmax + ue[j]*area2[j]*vmax + ub[j]*area2[j]*vmax + (pext[j]*area2[j]*vmax - pext[j-1]*area2[j-1]*v[j-1]);
      minfunc = -etot[0] + 0.5*lrminnl*pow(vmin,3.0)*area2[j] + (3.0/2.0)*press[j]*area2[j]*vmin + ue[j]*area2[j]*vmin + ub[j]*area2[j]*vmin + (pext[j]*area2[j]*vmin - pext[j-1]*area2[j-1]*v[j-1]);

      printf("Iterating %i for step %i: rhomid=%g, pbuoy=%g, area=%g, press=%g, v=%g, ekin=%g, eintt=%g, einte=%g, eintb=%g, etot=%g, work=%g, rfunc=%g, minfunc=%g, maxfunc=%g\n",iter, j, rhomid,pbuoy[j],area2[j],press[j],v[j],ekin[j],eintt[j],einte[j],eintb[j],etot[0],work[j],rfunc,minfunc,maxfunc);     
 
      if((rfunc == 0)||((lrmax-lrmin)/2.0<TOL)) {
	rsol = rhomid;
	printf("Found solution: r = %g; iter=%i, lrmax=%g, lrmin=%g\n",rhomid,iter,lrmax,lrmin); 
	break;
      } else {
	if(signum(rfunc)==signum(minfunc)) {
	  lrmin=log10(rhomid);
	} else {
	  lrmax=log10(rhomid);
	}
	
      }
    /*  printf("Currently on iteration number %i\n",iter); */
      iter++;
    }
    if(iter==IMAX) {
      /* printf("No solution found in %i iterations\n",iter); */
      rhotherm[j] = -999;
    } else {
      rhotherm[j] = rsol;
    }
    tnew[j] = press[j]*MU*MH/(QELEC*rhotherm[j]);
    psient[j] = rhotherm[j]*v[j]*area2[j] - rhotherm[j-1]*v[j-1]*area2[j-1];
    psil[j] = psient[j]/(rval[j]-rval[j-1]);
    fprintf(pout,"%g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",rval[j],area2[j],tnew[j],rhotherm[j],v[j],pbuoy[j],psient[j],psil[j],ekin[j],eintt[j],einte[j],eintb[j],etot[j],work[j]);   
  }


}
