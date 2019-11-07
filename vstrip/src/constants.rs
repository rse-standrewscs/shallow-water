use std::f64::consts::PI;

/*
! ==> Numerical parameters <==
integer,parameter:: ng=32,nz=4
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
*/

/// Number of vertical layers
pub const NG: usize = 32;

/// Inversion grid resolution in both x and y
/// (Note: the domain is a 2*pi periodic box.)
pub const _NZ: usize = 4;

/// Time step
pub const _DT: f64 = 1.0 / (2.0 * NG as f64);

/// Simulation duration
pub const _T_SIM: f64 = 25.0;

/// Grid data save time increment
pub const _TG_SAVE: f64 = 0.25;

/// Maximum pressure difference on convergence
pub const _TOLER: f64 = 0.000_000_001;

/// Constant Coriolis frequency
pub const _COF: f64 = 4.0 * PI;

/// Short-scale gravity wave speed
pub const _CGW: f64 = 2.0 * PI;

/// Mean fluid depth (conserved by mass conservation)
pub const _HBAR: f64 = 0.4;

/// This times f is the damping rate on wavenumber ng/2
pub const _C_DAMP: f64 = 10.0;

pub const _NNU: f64 = 3.0;
