module parameters

! Module containing all the modifiable parameters (except pi below).

! The number pi (*** non-modifiable ***):
double precision,parameter:: pi=3.141592653589793238462643383279502884197169399375105820974944592307816d0

! ==> Numerical parameters <==
integer,parameter:: ng=256,nz=32
double precision,parameter:: dt=1.d0/dble(ng),tsim=25.d0
double precision,parameter:: tgsave=0.25d0
double precision,parameter:: toler=1.d-9
! nz     : number of vertical layers
! ng     : inversion grid resolution in both x and y
!          (Note: the domain is a 2*pi periodic box.)
! dt     : time step (fixed)
! tsim   : total duration of the simulation
! tgsave : grid data save time increment
! toler  : maximum pressure difference on convergence

! ==> Physical parameters <==
double precision,parameter:: cof=4.d0*pi,cgw=2.d0*pi,hbar=0.4d0
double precision,parameter:: cdamp=10.d0,nnu=3
! cof    : Constant Coriolis frequency f
! cgw    : Short-scale gravity wave speed c
! hbar   : Mean fluid depth (conserved by mass conservation)
! cdamp  : This times f is the damping rate on wavenumber ng/2
!----------------------------------------------------------------

end module parameters
