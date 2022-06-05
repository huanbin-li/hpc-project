#include <petscksp.h>
#include <petscmath.h>
#include <math.h>



int main(int argc,char **args)
{
  double pi = acos(-1.0);
  Vec            x, u, f;          /* build the vecotr */
  Mat            A;                /* build the  matrix */
  PetscErrorCode ierr;             /* error checking */
  PetscInt       i, n=200, start=0, end=n, col[3], rstart,rend,nlocal,rank; /* n is region */
  PetscReal      dx, x, a, f0;
  PetscReal      p=1.0, c=1.0, k=1.0, t=0.0, u0=0.0, t0=2.0;   /* time step */
  PetscScalar    zero = 0.0, value[3];  /* u0 initial condition */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;/* initial petsc */
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr); /* read dt from command line */
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr); /* read n from command line */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr); /* set up for MPI */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "n = %d\n", n);CHKERRQ(ierr); /* print n */

  dx=1.0/n;
  a = k*dt/(p*c*dx*dx);
  
  ierr = PetscPrintf(PETSC_COMM_WORLD,"dx = %f\n",dx);CHKERRQ(ierr); /* check the dx */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"a = %f\n",a);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);/* create vector */
  ierr = VecSetSizes(x,PETSC_DECIDE,n+1);CHKERRQ(ierr); /* vector size*/
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);  /* enable option */
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);  /* copy the type and layout */
  ierr = VecDuplicate(x,&f);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr); /* set start and end */
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);  /* query the layout */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);  /* create matrix A */
  ierr = MatSetSizes(A,nlocal,nlocal,n+1,n+1);CHKERRQ(ierr); /* matrix size*/
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);  /* enable option */
  ierr = MatSetUp(A);CHKERRQ(ierr); 


  if (!rstart) 
  {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 1.0-2.0*a; value[1] = a;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  if (rend == n) 
  {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = a; value[1] = 1.0-2.0*a;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = a; value[1] = 1.0-2.0*a; value[2] = a;
  for (i=rstart; i<rend; i++) 
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  ierr = VecSet(u,zero);CHKERRQ(ierr);  /* set vector */
  if(rank == 0){
      for(int i=1; i<n; i++){   /* from 1 to n-1 point*/
        u0 = exp(i*dx);  
	      ierr = VecSetValues(u, 1, &i, &u0, INSERT_VALUES);CHKERRQ(ierr);
      }
  }
  
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr); /* Assemble the vector*/
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);

  ierr = VecSet(f,zero);CHKERRQ(ierr);
  if(rank == 0){
    for(int i = 1; i < n; i++){ 
      f0 = dt*sin(pi*i*dx); /* heat supply value */
      ierr = VecSetValues(f, 1, &i, &f, INSERT_VALUES);CHKERRQ(ierr); /* set value */
    }
  }

  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);  /* Assemble the vector*/
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);

  if(PetscAbsReal(t)<3.0){   /* set the caculate time */
        /* time advance*/
     ierr = MatMult(A,u,x);CHKERRQ(ierr); 
     ierr = VecAXPY(x,1.0,f);CHKERRQ(ierr); 

     ierr = VecSetValues(x, 1, &start, &zero, INSERT_VALUES);CHKERRQ(ierr); /* set value*/
     ierr = VecSetValues(x, 1, &end, &zero, INSERT_VALUES);CHKERRQ(ierr);
     ierr = VecAssemblyBegin(x);CHKERRQ(ierr); /* Assemble the vector*/
     ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
     ierr = VecCopy(x,u);CHKERRQ(ierr);  
     t = t+dt;

  }
	
  /* view the result */
  ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);  /* view the result */

  /* deallocate the vector and matirx */
  ierr = VecDestroy(&x);CHKERRQ(ierr);  
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();  /* finish */
  return ierr;
}
