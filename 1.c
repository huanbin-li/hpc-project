static char help[] = "Expilict Euler for 1D heat problem .\n\n";

#include <petscksp.h>
#include <petscmath.h>
#include <math.h>

#define pi acos(-1)   /* define pi */

int main(int argc,char **args)
{
  Vec            x, z, b;          /* build the vecotr */
  Mat            A;                /* build the  matrix */
  PetscErrorCode ierr;             /* error checking */
  PetscInt       i, n=200, start=0, end=n, col[3], rstart,rend,nlocal,rank; /* n is region */
  PetscReal      p=1.0, c=1.0, k=1.0, alpha, beta, dx, ix;/* pck is the physic parameter */
  PetscReal      dt=0.00001, t=0.0, u0=0.0;   /* time step */
  PetscScalar    zero = 0.0, value[3];  /* u0 initial condition */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;/* initial petsc */
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr); /* read dt from command line */
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr); /* read n from command line */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr); /* set up for MPI */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "n = %d\n", n);CHKERRQ(ierr); /* print n */

  dx=1.0/n;
  a = k/p/c;
  beta = a*dt/dx/dx;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"dx = %f\n",dx);CHKERRQ(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"beta = %f\n",beta);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n+1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);  
  ierr = VecDuplicate(x,&z);CHKERRQ(ierr); 
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);  

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);  
  ierr = MatSetSizes(A,nlocal,nlocal,n+1,n+1);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(A);CHKERRQ(ierr); 
  ierr = MatSetUp(A);CHKERRQ(ierr); 


