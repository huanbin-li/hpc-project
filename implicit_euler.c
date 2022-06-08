static char help[] = "Expilict Euler.\n\n";
#include <petscksp.h>
#include <petscmath.h>
#include <hdf5.h>
#include <petscviewerhdf5.h>

int main(int argc,char **args)
{
  double pi = acos(-1.0);
  Vec            x, u, f, para;          
  Mat            A;                
  PetscErrorCode ierr;            
  PetscInt       i, rstart,rend,nlocal,rank,tn;
  PetscInt       col[3];
  KSP            ksp; 
  PC             pc;    
           
  //n is the number of grids
  PetscInt       n=100, edge1=0, edge2=n, step=0;
  PetscReal      dx, a, f0, u0, t, tempdata;
  
  //dt is time step, t0 is the total time
  PetscReal      p=1.0, c=1.0, k=1.0, t0=2.0, dt=0.00001;   
  PetscScalar    zero = 0.0, one = 1.0, value[3], data0[3];
  
  //set the viewer
  PetscViewer    h5; 
  
  //initialize Petsc and enable MPI
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD, "n = %d\n", n);CHKERRQ(ierr);
  
  //read n,dt and restart step from command line
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);    
  ierr = PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);CHKERRQ(ierr);   
  ierr = PetscOptionsGetInt(NULL,NULL,"-step",&step,NULL);CHKERRQ(ierr);
  
  //set values for some parameters
  dx = 1.0/n;
  a = k*dt/(p*c*dx*dx);
  tn = floor(t0/dt);
  data0[0] = dx; 
  data0[1] = dt;
 
  
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"dx = %f\n",dx);CHKERRQ(ierr); 
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"a = %f\n",a);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"tn = %d\n",tn);CHKERRQ(ierr);
  
  //create vectors
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&para);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n+1);CHKERRQ(ierr); 
  ierr = VecSetSizes(para, PETSC_DECIDE, 3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr); 
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); 
  ierr = VecDuplicate(x,&f);CHKERRQ(ierr);
  

  
  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */
  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr); 
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr); 
  
  //Create matrix.  
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr); 
  ierr = MatSetSizes(A,nlocal,nlocal,n+1,n+1);CHKERRQ(ierr); 
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);  
  ierr = MatSetUp(A);CHKERRQ(ierr); 

 //set entries of the first and the last row
  if (!rstart) 
  {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 1.0+2.0*a; value[1] = -a;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  if (rend == n+1) 
  {
    rend = n;
    i    = n; col[0] = n-1; col[1] = n; value[0] = -a; value[1] = 1.0+2.0*a;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -a; value[1] = 1.0+2.0*a; value[2] = -a;
  for (i=rstart; i<rend; i++) 
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  //set the initial values for f
  ierr = VecSet(u,one);CHKERRQ(ierr);  
  if(rank == 0)
  {
      for(int i=1; i<n; i++)
	  {   
        u0 = exp(i*dx);  
	      ierr = VecSetValues(u, 1, &i, &u0, INSERT_VALUES);CHKERRQ(ierr);
      }
  }
  //assemble the vector
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr); 
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  //set the values for f
  ierr = VecSet(f,one);CHKERRQ(ierr);
  if(rank == 0)
  {
      for(int i = 1; i < n; i++)
	  { 
        f0 = dt*sin(pi*i*dx); 
        ierr = VecSetValues(f, 1, &i, &f0, INSERT_VALUES);CHKERRQ(ierr);
      }
  }
  //assemble the vector
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr); 
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);

  //ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  //create linear solver
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  //set operators
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);    
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  //set different tolerances
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr); 
  //set run time options 
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr); 
  
  //if input step=1 from command line,read data from output file of last run
  if(step > 0)
  {
  	ierr = PetscViewerCreate(PETSC_COMM_WORLD,&h5);CHKERRQ(ierr);  
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"out.h5", FILE_MODE_READ, &h5);CHKERRQ(ierr);    
    ierr = PetscObjectSetName((PetscObject) para, "para");CHKERRQ(ierr); 
    ierr = PetscObjectSetName((PetscObject) u, "temp");CHKERRQ(ierr); 
    //load datas from HDF5 file
    ierr = VecLoad(u, h5);CHKERRQ(ierr);
	ierr = VecLoad(para, h5);CHKERRQ(ierr);      
      
    //load time and grid number values from HDF5 file
    for(int i = 0; i < 2; i++)
	{ 
	    /*if (i==0)
	    {
	       ierr = VecGetValues(para,1,&i,&dx);CHKERRQ(ierr); 	
	    }
	    if (i==1)
	    {
	       ierr = VecGetValues(para,1,&i,&dt);CHKERRQ(ierr); 	
	    }
	    */
	    if (i==2)
	    {
	       ierr = VecGetValues(para,1,&i,&t);CHKERRQ(ierr); 
		   ierr = PetscPrintf(PETSC_COMM_WORLD, "restart from t = %d\n", t);CHKERRQ(ierr);
	    }
    }
    ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr); 
    
    /*index=0;   
    ierr = VecGetValues(temp,1,&index,&dx);CHKERRQ(ierr);    
    index += 1;    
    ierr = VecGetValues(temp,1,&index,&dt);CHKERRQ(ierr);    
    index += 1;  
    ierr = VecGetValues(temp,1,&index,&t);CHKERRQ(ierr);   
    index= 0; 
	*/  
  }
  
  //do the iteration
  for(int i = 0; i < tn; i++)
  {    
     
     ierr = VecAXPY(u, 1.0, f);CHKERRQ(ierr); 
	 //solve the linear equation  
     ierr = KSPSolve(ksp,u,x);CHKERRQ(ierr);    
     //boundary conditions
     ierr = VecSetValues(x, 1, &edge1, &zero, INSERT_VALUES);CHKERRQ(ierr); 
     ierr = VecSetValues(x, 1, &edge2, &zero, INSERT_VALUES);CHKERRQ(ierr);
     
     ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
     ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
     ierr = VecCopy(x,u);CHKERRQ(ierr);  
     
     if ((i%10)== 0)
     {
     	ierr = PetscViewerCreate(PETSC_COMM_WORLD,&h5);CHKERRQ(ierr); 
     	//initialize
     	//ierr = VecSet(para,zero);CHKERRQ(ierr); 
     	data0[2] = i*dt;   
        //write the values into the array of parameter
        for(i = 0; i < 3; i++)
        {   
          tempdata = data0[i];   
          ierr = VecSetValues(para,1,&i,&tempdata,INSERT_VALUES);CHKERRQ(ierr);   
        }
        ierr = VecAssemblyBegin(para);CHKERRQ(ierr);   
        ierr = VecAssemblyEnd(para);CHKERRQ(ierr);  
        //view the temperature data and parameters
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"out.h5", FILE_MODE_WRITE, &h5);CHKERRQ(ierr);    
        ierr = PetscObjectSetName((PetscObject) u, "temp");CHKERRQ(ierr);   
        ierr = PetscObjectSetName((PetscObject) para, "parameter");CHKERRQ(ierr);   
        ierr = VecView(para, h5);CHKERRQ(ierr);    
        ierr = VecView(u, h5);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&h5);CHKERRQ(ierr);
	 }
  }
	
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  //deallocate the vectors and matrix
  ierr = VecDestroy(&x);CHKERRQ(ierr);  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize(); 
  return ierr;
}
