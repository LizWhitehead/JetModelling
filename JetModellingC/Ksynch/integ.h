/* exports the numerical integration routines from Numerical Recipes */
/* midin1, midin2 and midin3 are the integrating subroutines (see
   Num. Rec. p. 118.) midin1 works in 1/x space and so can integrate
   to plus or minus infinity, but not across 0. midin2 and 3 can go
   across 0 but not to infinity. They all use the extended midpoint
   rule, i.e. the integrand doesn't have to be evaluated at the
   endpoints. They use static storage space and so must not be called
   recursively */

/* extern double polint(double [], double [], int, double, double *, double *); */

extern double qromo1(double (), double, double, void ());

extern void midin1();

extern void midin2();

extern void midin3();

#define FAILTAG -666.666
#define EPS 1.0e-5
#define INFIN  1.0e30

/* if qromo1 fails it returns FAILTAG. EPS is the precision of
   integration. INFIN is the infinity of integration */
