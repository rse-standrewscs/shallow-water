!#########################################################################
!          The Horizontally Doubly-Periodic Three-Dimensional
!         Non-Hydrostatic Shallow-Water Pseudo-Spectral Method
!#########################################################################

!       Code adapted from ~/hydra/ps/plane/sw/swps codes in March 2019
!       by D G Dritschel @ St Andrews.

!       This code simulates the unforced non-hydrostatic shallow-water 
!       equations in variables (ql,delta,gamma), where ql is the linearised 
!       potential vorticity anomaly, delta is the velocity divergence, 
!       and gamma is the acceleration divergence (called ageostrophic 
!       vorticity).  Note: ql = zeta - f*rho', where zeta is the relative
!       vorticity, f is the Coriolis frequency and rho' is the dimensionless
!       layer thickness anomaly.

!       The full algorithm consists of the following modules:

!       nhswps.f90    : This source - main program to evolve fields;
!       parameters.f90: User defined parameters for a simulation;
!       constants.f90 : Fixed constants used throughout the other modules;
!       spectral.f90  : Fourier transform common storage and routines;

!----------------------------------------------------------------------------
program nhswps

 !Import contants, parameters and common arrays from spectral module:
use spectral

implicit none

 !Define common space:

 !Velocity field (physical):
double precision:: u(ng,ng,0:nz),v(ng,ng,0:nz),w(ng,ng,0:nz)

 !Layer heights and their x & y derivatives (physical):
double precision:: z(ng,ng,0:nz),zx(ng,ng,0:nz),zy(ng,ng,0:nz)

 !Dimensionless layer thickness anomaly and inverse thickness (physical):
double precision:: r(ng,ng,0:nz),ri(ng,ng,0:nz)

 !A = grad{u*rho'_theta} (spectral):
double precision:: aa(ng,ng,0:nz)

 !Relative vertical vorticity component (physical):
double precision:: zeta(ng,ng,0:nz)

 !Non-hydrostatic pressure (p_n) and its first derivative wrt theta:
double precision:: pn(ng,ng,0:nz),dpn(ng,ng,0:nz)

 !Non-hydrostatic pressure (p_n) in spectral space (called ps):
double precision:: ps(ng,ng,0:nz)

 !Prognostic fields q_l, delta and gamma (spectral):
double precision:: qs(ng,ng,0:nz),ds(ng,ng,0:nz),gs(ng,ng,0:nz)

 !Time:
double precision:: t

 !Number of time steps between field saves and related indices:
integer:: ngsave,itime,jtime

 !Logical for use in calling inversion routine:
logical:: ggen

!---------------------------------------------------------
 !Define fixed arrays and constants and read initial data:
call initialise

!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 !Start the time loop:
do while (t .le. tsim)

   !Save data periodically:
  itime=nint(t/dt)
  jtime=itime/ngsave
  if (ngsave*jtime .eq. itime) then
     !Invert PV, divergence and acceleration divergence to obtain the
     !dimensionless layer thickness anomaly and horizontal velocity, 
     !as well as the relative vertical vorticity (see spectral.f90):
    call main_invert(qs,ds,gs,r,u,v,zeta)
     !Note: qs, ds & gs are in spectral space while 
     !      r, u, v and zeta are in physical space.
     !Next find the non-hydrostatic pressure (pn), layer heights (z)
     !and vertical velocity (w):
    call psolve
     !Save field data:
    call savegrid(jtime+1)
    ggen=.false.
  else
    ggen=.true.
  endif
   !ggen is used to indicate if calling inversion is needed in advance below

   !Advect flow from time t to t + dt:
  call advance(ggen)
  
enddo
!End of time loop
!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

 !Possibly save final data:
itime=nint(t/dt)
jtime=itime/ngsave
if (ngsave*jtime .eq. itime) then
  call main_invert(qs,ds,gs,r,u,v,zeta)
  call psolve
  call savegrid(jtime+1)
endif

 !Close all files:
call finalise


 !Internal subroutine definitions (inherit global variables):
!::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

contains

!=======================================================================

subroutine initialise

! Routine initialises fixed constants and arrays, and reads in
! input files, opens output files ready for writing to. 

implicit none

integer:: iz

!----------------------------------------------------------------------
 !Initialise inversion constants and arrays:
call init_spectral

!----------------------------------------------------------------------
 !Read linearised PV anomaly and convert to spectral space as qs:
open(11,file='qq_init.r8',form='unformatted', &
    & access='direct',status='old',recl=2*ntbytes)
read(11,rec=1) t,z
close(11)

call ptospc3d(z,qs,0,nz)
 !Ensure horizontal average qs is zero in each layer:
qs(1,1,:)=zero

!----------------------------------------------------------------------
 !Read divergence and convert to spectral space as ds:
open(11,file='dd_init.r8',form='unformatted', &
    & access='direct',status='old',recl=2*ntbytes)
read(11,rec=1) t,z
close(11)

call ptospc3d(z,ds,0,nz)
 !Horizontal average ds must be zero in each layer:
ds(1,1,:)=zero

!----------------------------------------------------------------------
 !Read acceleration divergence and convert to spectral space as gs:
open(11,file='gg_init.r8',form='unformatted', &
    & access='direct',status='old',recl=2*ntbytes)
read(11,rec=1) t,z
close(11)

call ptospc3d(z,gs,0,nz)
 !Horizontal average gs must be zero in each layer:
gs(1,1,:)=zero

!----------------------------------------------------------------------
 !Spectrally-truncate all fields for use in de-aliasing:
do iz=0,nz
  qs(:,:,iz)=filt*qs(:,:,iz)
  ds(:,:,iz)=filt*ds(:,:,iz)
  gs(:,:,iz)=filt*gs(:,:,iz)
enddo

 !Obtain initial dimensionless layer thickness anomaly (r), 
 !horizontal velocity (u,v) and relative vertical vorticity (zeta):
call main_invert(qs,ds,gs,r,u,v,zeta)
 !Note: qs, ds & gs are in spectral space while 
 !      r, u, v and zeta are in physical space.

 !Initialise non-hydrostatic pressure (needed in first call to psolve)
pn=zero

 !Define number of time steps between grid and contour saves:
ngsave=nint(tgsave/dt)
 !*** WARNING: tgsave should be an integer multiple of dt

!--------------------------------------
 !Open all plain text diagnostic files:
open(16,file='ecomp.asc',status='replace')
open(17,file='monitor.asc',status='replace')

 !Open file for 1d vorticity & divergence spectra:
open(51,file='spectra.asc',status='replace')

 !Open files for selected 3d fields:
open(31,file='3d/ql.r4',form='unformatted',access='direct', &
                      status='replace',recl=ntbytes)
open(32,file= '3d/d.r4',form='unformatted',access='direct', &
                      status='replace',recl=ntbytes)
open(33,file= '3d/g.r4',form='unformatted',access='direct', &
                      status='replace',recl=ntbytes)
open(34,file= '3d/r.r4',form='unformatted',access='direct', &
                      status='replace',recl=ntbytes)
open(35,file= '3d/w.r4',form='unformatted',access='direct', &
                      status='replace',recl=ntbytes)
open(36,file='3d/pn.r4',form='unformatted',access='direct', &
                      status='replace',recl=ntbytes)

 !Open files for selected vertically-averaged fields:
open(41,file='2d/q.r4',form='unformatted',access='direct', &
                         status='replace',recl=nhbytes)
open(42,file='2d/d.r4',form='unformatted',access='direct', &
                         status='replace',recl=nhbytes)
open(43,file='2d/g.r4',form='unformatted',access='direct', &
                         status='replace',recl=nhbytes)
open(44,file='2d/h.r4',form='unformatted',access='direct', &
                         status='replace',recl=nhbytes)
open(45,file='2d/zeta.r4',form='unformatted',access='direct', &
                         status='replace',recl=nhbytes)

return
end subroutine

!=======================================================================

subroutine advance(ggen)

! Advances fields from time t to t+dt using an iterative implicit 
! method of the form
!
! (F^{n+1}-F^n)/dt = L[(F^{n+1}-F^n)/2] + N[(F^{n+1}-F^n)/2]
!
! for a field F, where n refers to the time level, L refers to
! the linear source terms, and N refers to the nonlinear source
! terms.  We start with a guess for F^{n+1} in N and iterate 
! niter times (see parameter statement below).

implicit none

 !Passed variable:
logical:: ggen

 !Local variables:
integer,parameter:: niter=2

 !Spectral fields needed in time stepping:
double precision:: qsi(ng,ng,0:nz),qsm(ng,ng,0:nz),sqs(ng,ng,0:nz)
double precision:: dsi(ng,ng,0:nz),sds(ng,ng,0:nz),nds(ng,ng,0:nz)
double precision:: gsi(ng,ng,0:nz),sgs(ng,ng,0:nz),ngs(ng,ng,0:nz)
double precision:: wka(ng,ng),wkb(ng,ng)

 !Other local quantities:
integer:: iz,iter

!------------------------------------------------------------------
 !Invert PV and compute velocity at current time level, say t=t^n:
if (ggen) then
  call main_invert(qs,ds,gs,r,u,v,zeta)
  call psolve
endif

 !If ggen is false, main_invert and psolve were called previously 
 !at this time level.

 !Save various diagnostics each time step:
call diagnose

!------------------------------------------------------------------
 !Start with a guess for F^{n+1} for all fields:

 !Calculate the source terms (sqs,sds,sgs) for linearised PV (qs), 
 !divergence (ds) and acceleration divergence (gs):
call source(sqs,sds,sgs)

 !Update PV field:
qsi=qs
qsm=qs+dt4*sqs
do iz=0,nz
  qs(:,:,iz)=diss*(qsm(:,:,iz)+dt4*sqs(:,:,iz))-qsi(:,:,iz)
enddo

 !Update divergence and acceleration divergence:
dsi=ds
gsi=gs
nds=sds+dt4i*dsi
ngs=sgs+dt4i*gsi
sds=nds+sds          !2*N_tilde_delta
sgs=ngs+sgs          !2*N_tilde_gamma
wka=zero
wkb=zero
do iz=0,nz
  ds(:,:,iz)=sgs(:,:,iz)+rdis*sds(:,:,iz)          !2*T_tilde_delta
  wka=wka+weight(iz)*ds(:,:,iz)
  wkb=wkb+weight(iz)*sds(:,:,iz)
enddo
wka=fope*wka         !fope = F operator
wkb=c2g2*wkb         !c2g2 = c^2*Lap operator
do iz=0,nz
  ds(:,:,iz)=simp*(ds(:,:,iz)-wka)-dsi(:,:,iz)     !simp = (R^2 + f^2)^{-1}
  gs(:,:,iz)=wkb-fsq*sds(:,:,iz)+rdis*sgs(:,:,iz)  !2*T_tilde_gamma
enddo
wka=zero
do iz=0,nz
  wka=wka+weight(iz)*gs(:,:,iz)
enddo
wka=fope*wka         !fope = F operator in paper
do iz=0,nz
  gs(:,:,iz)=simp*(gs(:,:,iz)-wka)-gsi(:,:,iz)     !simp = (R^2 + f^2)^{-1}
enddo

!------------------------------------------------------------------
 !Iterate to improve estimates of F^{n+1}:
do iter=1,niter
   !Perform inversion at t^{n+1} from estimated quantities:
  call main_invert(qs,ds,gs,r,u,v,zeta)

   !Compute pressure, etc:
  call psolve

   !Calculate the source terms (sqs,sds,sgs) for linearised PV (qs), 
   !divergence (ds) and acceleration divergence (gs):
  call source(sqs,sds,sgs)

   !Update PV field:
  do iz=0,nz
    qs(:,:,iz)=diss*(qsm(:,:,iz)+dt4*sqs(:,:,iz))-qsi(:,:,iz)
  enddo
 
   !Update divergence and acceleration divergence:
  sds=nds+sds          !2*N_tilde_delta
  sgs=ngs+sgs          !2*N_tilde_gamma
  wka=zero
  wkb=zero
  do iz=0,nz
    ds(:,:,iz)=sgs(:,:,iz)+rdis*sds(:,:,iz)          !2*T_tilde_delta
    wka=wka+weight(iz)*ds(:,:,iz)
    wkb=wkb+weight(iz)*sds(:,:,iz)
  enddo
  wka=fope*wka         !fope = F operator
  wkb=c2g2*wkb         !c2g2 = c^2*Lap operator
  do iz=0,nz
    ds(:,:,iz)=simp*(ds(:,:,iz)-wka)-dsi(:,:,iz)     !simp = (R^2 + f^2)^{-1}
    gs(:,:,iz)=wkb-fsq*sds(:,:,iz)+rdis*sgs(:,:,iz)  !2*T_tilde_gamma
  enddo
  wka=zero
  do iz=0,nz
    wka=wka+weight(iz)*gs(:,:,iz)
  enddo
  wka=fope*wka         !fope = F operator in paper
  do iz=0,nz
    gs(:,:,iz)=simp*(gs(:,:,iz)-wka)-gsi(:,:,iz)     !simp = (R^2 + f^2)^{-1}
  enddo
enddo

 !Advance time:
t=t+dt

return
end subroutine

!=============================================================

subroutine psolve
! Solves for the nonhydrostatic part of the pressure (pn) given
! the velocity field (u,v,w) together with r = rho'_theta and
! z = theta + int_0^theta{rho'_theta(s)ds}.

implicit none

 !Local variables:
integer,parameter:: nitmax=100
 !nitmax: maximum number of iterations allowed before stopping

! Constant part of the pressure source:
double precision:: sp0(ng,ng,0:nz)

! Arrays used for pressure inversion (these depend on rho'_theta only):
double precision:: sigx(ng,ng,0:nz),sigy(ng,ng,0:nz)
double precision:: cpt1(ng,ng,0:nz),cpt2(ng,ng,0:nz)

! Physical space arrays:
double precision:: pna(ng,ng,0:nz)
double precision:: dpdt(ng,ng),d2pdxt(ng,ng),d2pdyt(ng,ng),d2pdt2(ng,ng)
double precision:: wkp(ng,ng),wkq(ng,ng)
! Spectral space arrays (all work arrays):
double precision:: sp(ng,ng,0:nz),gg(ng,ng,0:nz)
double precision:: wka(ng,ng),wkb(ng,ng),wkc(ng,ng),wkd(ng,ng)
! Other scalars:
double precision:: errp
integer:: iz,iter

!---------------------------------------------------------------
! Calculate 1/(1+rho'_theta) and de-aliase:
ri=one/(one+r)
call deal3d(ri)

! Calcuate layer heights z and z_x & z_y, vertical velocity w
! and A = grad(u*rho'_theta):
call vertical

! Define constant coefficients in pressure inversion:
call coeffs(sigx,sigy,cpt1,cpt2)

! Define constant part of the pressure source (sp0):
call cpsource(sp0)

! Solve for the pressure using previous solution as first guess:
pna=pn

!################################################################
! Begin iteration to find (non-hydrostatic part of the) pressure:
errp=one
iter=0
do while (errp > toler .and. iter < nitmax) 

  ! Get spectral coefficients for pressure:
  call ptospc3d(pn,ps,0,nz-1)
  ps(:,:,nz)=zero

  ! Compute pressure derivatives needed in the non-constant part of the
  ! source S_1 and add to S_0 (in sp0) to form total source S (sp):

  ! Lower boundary at iz = 0 (use dp/dtheta = 0):
  ! d^2p/dtheta^2:
  wkd=(two*ps(:,:,0)-five*ps(:,:,1)+four*ps(:,:,2)-ps(:,:,3))*dzisq
  ! Return to physical space:
  call spctop(ng,ng,wkd,d2pdt2,xfactors,yfactors,xtrig,ytrig)
  ! Total source:
  wkp=sp0(:,:,0)+cpt2(:,:,0)*d2pdt2
  ! Transform to spectral space for inversion below:
  call ptospc(ng,ng,wkp,wka,xfactors,yfactors,xtrig,ytrig)
  sp(:,:,0)=wka

  ! Interior grid points:
  do iz=1,nz-1
    wkq=d2pdt2
    wka=(ps(:,:,iz+1)-ps(:,:,iz-1))*hdzi
    wkd=(ps(:,:,iz+1)-two*ps(:,:,iz)+ps(:,:,iz-1))*dzisq
    ! Calculate x & y derivatives of dp/dtheta:
    call xderiv(ng,ng,hrkx,wka,wkb)
    call yderiv(ng,ng,hrky,wka,wkc)
    ! Return to physical space:
    call spctop(ng,ng,wka,dpdt,xfactors,yfactors,xtrig,ytrig)
    call spctop(ng,ng,wkb,d2pdxt,xfactors,yfactors,xtrig,ytrig)
    call spctop(ng,ng,wkc,d2pdyt,xfactors,yfactors,xtrig,ytrig)
    call spctop(ng,ng,wkd,d2pdt2,xfactors,yfactors,xtrig,ytrig)
    ! Total source:
    wkp=sp0(:,:,iz)+sigx(:,:,iz)*d2pdxt+sigy(:,:,iz)*d2pdyt &
                   +cpt2(:,:,iz)*d2pdt2+cpt1(:,:,iz)*dpdt
    ! Transform to spectral space for inversion below:
    call ptospc(ng,ng,wkp,wka,xfactors,yfactors,xtrig,ytrig)
    sp(:,:,iz)=wka
  enddo

  ! Upper boundary at iz = nz (use p = 0):
  ! Extrapolate to find first and second derivatives there:
  dpdt=dpdt+dz2*(three*d2pdt2-wkq)
  d2pdt2=two*d2pdt2-wkq
  wkp=dpdt
  call ptospc(ng,ng,wkp,wka,xfactors,yfactors,xtrig,ytrig)
  ! Calculate x & y derivatives of dp/dtheta:
  call xderiv(ng,ng,hrkx,wka,wkb)
  call yderiv(ng,ng,hrky,wka,wkc)
  ! Return to physical space:
  call spctop(ng,ng,wkb,d2pdxt,xfactors,yfactors,xtrig,ytrig)
  call spctop(ng,ng,wkc,d2pdyt,xfactors,yfactors,xtrig,ytrig)
  ! Total source:
  wkp=sp0(:,:,nz)+sigx(:,:,nz)*d2pdxt+sigy(:,:,nz)*d2pdyt &
                 +cpt2(:,:,nz)*d2pdt2+cpt1(:,:,nz)*dpdt

  ! Transform to spectral space for inversion below:
  call ptospc(ng,ng,wkp,wka,xfactors,yfactors,xtrig,ytrig)
  sp(:,:,nz)=wka

  !----------------------------------------------------------
  ! Solve tridiagonal problem for pressure in spectral space:
  gg(:,:,0)=f13*sp(:,:,0)+f16*sp(:,:,1)
  do iz=1,nz-1
    gg(:,:,iz)=f112*(sp(:,:,iz-1)+sp(:,:,iz+1))+f56*sp(:,:,iz)
  enddo

  ps(:,:,0)=gg(:,:,0)*htdv(:,:,0)
  do iz=1,nz-1
    ps(:,:,iz)=(gg(:,:,iz)-ap*ps(:,:,iz-1))*htdv(:,:,iz)
  enddo
  do iz=nz-2,0,-1
    ps(:,:,iz)=etdv(:,:,iz)*ps(:,:,iz+1)+ps(:,:,iz)
  enddo
  ps(:,:,nz)=zero

  ! Transform to physical space:
  call spctop3d(ps,pn,0,nz-1)
  pn(:,:,nz)=zero

  ! Monitor convergence:
  errp=sqrt(sum((pn-pna)**2)/(sum(pna**2)+1.d-20))

  ! Stop if not converging:
  if (iter > 0 .and. errp > one) then
    write(*,*) ' Pressure error too large!'
    write(*,*) ' Final pressure error = ',errp
    write(*,*) ' *** Stopping!'
    stop
  endif

  iter=iter+1

  ! Reset pna:
  pna=pn

enddo

if (iter >= nitmax) then
  write(*,*) ' Exceeded maximum number of iterations to find pressure!'
  write(*,*) ' Final pressure error = ',errp
  write(*,*) ' *** Stopping!'
endif

!################################################################

! Past this point, we have converged!

! Calculate 1st derivative of pressure using 4th-order compact differences:
do iz=1,nz-1
  gg(:,:,iz)=(ps(:,:,iz+1)-ps(:,:,iz-1))*hdzi
enddo
gg(:,:,nz)=dz6*sp(:,:,nz)-ps(:,:,nz-1)*dzi

gg(:,:,1)=gg(:,:,1)*htd1(1)
do iz=2,nz-1
  gg(:,:,iz)=(gg(:,:,iz)-f16*gg(:,:,iz-1))*htd1(iz)
enddo
gg(:,:,nz)=(gg(:,:,nz)-f13*gg(:,:,nz-1))*htd1(nz)
do iz=nz-1,1,-1
  gg(:,:,iz)=etd1(iz)*gg(:,:,iz+1)+gg(:,:,iz)
enddo

! Transform to physical space:
call spctop3d(gg,dpn,1,nz)

return
end subroutine

!=============================================================

subroutine cpsource(sp0)
! Finds the part of the pressure source which does not vary
! in the iteration to find the pressure.

implicit none

 !Passed variable:

! Constant part of the pressure source:
double precision:: sp0(ng,ng,0:nz)

 !Local variables:

! Physical space arrays:
double precision:: ut(ng,ng,0:nz),vt(ng,ng,0:nz),wt(ng,ng,0:nz)
double precision:: ux(ng,ng),uy(ng,ng),vx(ng,ng),vy(ng,ng)
double precision:: wx(ng,ng),wy(ng,ng)
double precision:: hsrc(ng,ng),wkp(ng,ng),wkq(ng,ng),wkr(ng,ng)
! Spectral space arrays (all work arrays):
double precision:: wka(ng,ng),wkb(ng,ng)
! Other scalars:
integer:: iz

!---------------------------------------------------------------
! Calculate part which is independent of z, -g*Lap_h{h}:
wkp=z(:,:,nz)
! wkp = h; Fourier transform to spectral space:
call ptospc(ng,ng,wkp,wka,xfactors,yfactors,xtrig,ytrig)
! Apply -g*Lap_h operator:
wka=glap*wka
! Return to physical space:
call spctop(ng,ng,wka,hsrc,xfactors,yfactors,xtrig,ytrig)
! hsrc contains -g*Lap{h} in physical space.

!---------------------------------------------------------------
! Calculate u_theta, v_theta & w_theta:

! Lower boundary (use higher order formula):
ut(:,:,0)=hdzi*(four*u(:,:,1)-three*u(:,:,0)-u(:,:,2))
vt(:,:,0)=hdzi*(four*v(:,:,1)-three*v(:,:,0)-v(:,:,2))
wt(:,:,0)=hdzi*(four*w(:,:,1)-three*w(:,:,0)-w(:,:,2))

! Interior (centred differencing):
do iz=1,nz-1
  ut(:,:,iz)=hdzi*(u(:,:,iz+1)-u(:,:,iz-1))
  vt(:,:,iz)=hdzi*(v(:,:,iz+1)-v(:,:,iz-1))
  wt(:,:,iz)=hdzi*(w(:,:,iz+1)-w(:,:,iz-1))
enddo

! Upper boundary (use higher order formula):
ut(:,:,nz)=hdzi*(three*u(:,:,nz)+u(:,:,nz-2)-four*u(:,:,nz-1))
vt(:,:,nz)=hdzi*(three*v(:,:,nz)+v(:,:,nz-2)-four*v(:,:,nz-1))
wt(:,:,nz)=hdzi*(three*w(:,:,nz)+w(:,:,nz-2)-four*w(:,:,nz-1))

!---------------------------------------------------------------
! Loop over layers and build up source, sp0:

! iz = 0 is much simpler as z = w = 0 there:
! Calculate u_x, u_y, v_x & v_y:
wkq=u(:,:,0)
call ptospc(ng,ng,wkq,wka,xfactors,yfactors,xtrig,ytrig)
call xderiv(ng,ng,hrkx,wka,wkb)
call spctop(ng,ng,wkb,ux,xfactors,yfactors,xtrig,ytrig)
call yderiv(ng,ng,hrky,wka,wkb)
call spctop(ng,ng,wkb,uy,xfactors,yfactors,xtrig,ytrig)
wkq=v(:,:,0)
call ptospc(ng,ng,wkq,wka,xfactors,yfactors,xtrig,ytrig)
call xderiv(ng,ng,hrkx,wka,wkb)
call spctop(ng,ng,wkb,vx,xfactors,yfactors,xtrig,ytrig)
call yderiv(ng,ng,hrky,wka,wkb)
call spctop(ng,ng,wkb,vy,xfactors,yfactors,xtrig,ytrig)
wkq=ri(:,:,0)*wt(:,:,0)
call deal2d(wkq)
sp0(:,:,0)=hsrc+cof*zeta(:,:,0)+two*(ux*vy-uy*vx+wkq*(ux+vy))

! Remaining layers:
do iz=1,nz
  ! Calculate u_x, u_y, v_x, v_y, w_x, w_y:
  wkq=u(:,:,iz)
  call ptospc(ng,ng,wkq,wka,xfactors,yfactors,xtrig,ytrig)
  call xderiv(ng,ng,hrkx,wka,wkb)
  call spctop(ng,ng,wkb,ux,xfactors,yfactors,xtrig,ytrig)
  call yderiv(ng,ng,hrky,wka,wkb)
  call spctop(ng,ng,wkb,uy,xfactors,yfactors,xtrig,ytrig)
  wkq=v(:,:,iz)
  call ptospc(ng,ng,wkq,wka,xfactors,yfactors,xtrig,ytrig)
  call xderiv(ng,ng,hrkx,wka,wkb)
  call spctop(ng,ng,wkb,vx,xfactors,yfactors,xtrig,ytrig)
  call yderiv(ng,ng,hrky,wka,wkb)
  call spctop(ng,ng,wkb,vy,xfactors,yfactors,xtrig,ytrig)
  wkq=w(:,:,iz)
  call ptospc(ng,ng,wkq,wka,xfactors,yfactors,xtrig,ytrig)
  call xderiv(ng,ng,hrkx,wka,wkb)
  call spctop(ng,ng,wkb,wx,xfactors,yfactors,xtrig,ytrig)
  call yderiv(ng,ng,hrky,wka,wkb)
  call spctop(ng,ng,wkb,wy,xfactors,yfactors,xtrig,ytrig)
  ! Calculate pressure source:
  wkp=vt(:,:,iz)*zx(:,:,iz)-ut(:,:,iz)*zy(:,:,iz)
  call deal2d(wkp)
  wkq=uy*vt(:,:,iz)-ut(:,:,iz)*vy
  call deal2d(wkq)
  wkr=ut(:,:,iz)*vx-ux*vt(:,:,iz)
  call deal2d(wkr)
  wkq=wkq*zx(:,:,iz)+wkr*zy(:,:,iz)+ &
      (ux+vy)*wt(:,:,iz)-wx*ut(:,:,iz)-wy*vt(:,:,iz)
  call deal2d(wkq)
  sp0(:,:,iz)=hsrc+cof*(zeta(:,:,iz)-ri(:,:,iz)*wkp)+ &
              two*(ux*vy-uy*vx+ri(:,:,iz)*wkq)
enddo

return
end subroutine

!=============================================================

subroutine coeffs(sigx,sigy,cpt1,cpt2)
! Calculates the fixed coefficients used in the pressure iteration.

implicit none

 !Passed variables:
double precision:: sigx(ng,ng,0:nz),sigy(ng,ng,0:nz)
double precision:: cpt1(ng,ng,0:nz),cpt2(ng,ng,0:nz)

 !Local variables:
double precision:: wkp(ng,ng)
double precision:: wka(ng,ng)
integer:: iz

!---------------------------------------------------------------
! Compute sigx and sigy and de-alias:
sigx=ri*zx
sigy=ri*zy
call deal3d(sigx)
call deal3d(sigy)

! Compute cpt2 and de-alias:
cpt2=one-ri**2-sigx**2-sigy**2
call deal3d(cpt2)

!-------------------------------------------------------------------
! Calculate 0.5*d(cpt2)/dtheta + div(sigx,sigy) and store in cpt1:

! Lower boundary (use higher order formula):
cpt1(:,:,0)=qdzi*(four*cpt2(:,:,1)-three*cpt2(:,:,0)-cpt2(:,:,2))
! qdzi=1/(4*dz) is used since 0.5*d/dtheta is being computed.

! Interior (centred differencing):
do iz=1,nz-1
  call divs(sigx(:,:,iz),sigy(:,:,iz),wka)
  call spctop(ng,ng,wka,wkp,xfactors,yfactors,xtrig,ytrig)
  cpt1(:,:,iz)=qdzi*(cpt2(:,:,iz+1)-cpt2(:,:,iz-1))+wkp
enddo

! Upper boundary (use higher order formula):
call divs(sigx(:,:,nz),sigy(:,:,nz),wka)
call spctop(ng,ng,wka,wkp,xfactors,yfactors,xtrig,ytrig)
cpt1(:,:,nz)=qdzi*(three*cpt2(:,:,nz)+cpt2(:,:,nz-2)-four*cpt2(:,:,nz-1))+wkp

! Re-define sigx and sigy to include a factor of 2:
sigx=two*sigx
sigy=two*sigy

return
end subroutine

!===================================================================

subroutine vertical
! Calculates layer heights (z), as well as dz/dx & dz/dy (zx & zy),
! the vertical velocity (w), and the A = grad{u*rho'_theta} (aa).

implicit none

 !Local variables:
! Physical space arrays:
double precision:: rsrc(ng,ng,0:nz)
double precision:: wkq(ng,ng)
! Spectral space arrays (all work arrays):
double precision:: wka(ng,ng),wkb(ng,ng),wkc(ng,ng)
! Other scalars:
integer:: iz

!---------------------------------------------------------------
! Only need to consider iz > 0 as z = w = 0 for iz = 0:

! Find z by trapezoidal integration of rho_theta (integrate over
! rho'_theta then add theta to the result):
z(:,:,1)=dz2*(r(:,:,0)+r(:,:,1))
do iz=1,nz-1
  z(:,:,iz+1)=z(:,:,iz)+dz2*(r(:,:,iz)+r(:,:,iz+1))
enddo

do iz=1,nz
  ! Add on theta (a linear function) to complete definition of z:
  z(:,:,iz)=z(:,:,iz)+theta(iz)

  ! Calculate z_x & z_y:
  wkq=z(:,:,iz)
  call ptospc(ng,ng,wkq,wka,xfactors,yfactors,xtrig,ytrig)
  call xderiv(ng,ng,hrkx,wka,wkb)
  call spctop(ng,ng,wkb,wkq,xfactors,yfactors,xtrig,ytrig)
  zx(:,:,iz)=wkq
  call yderiv(ng,ng,hrky,wka,wkb)
  call spctop(ng,ng,wkb,wkq,xfactors,yfactors,xtrig,ytrig)
  zy(:,:,iz)=wkq
enddo

! Calculate A = grad{u*rho'_theta} (spectral):
do iz=0,nz
  ! Calculate (u*rho'_theta)_x:
  wkq=u(:,:,iz)*r(:,:,iz)
  call ptospc(ng,ng,wkq,wka,xfactors,yfactors,xtrig,ytrig)
  call xderiv(ng,ng,hrkx,wka,wkb)

  ! Calculate (v*rho'_theta)_y:
  wkq=v(:,:,iz)*r(:,:,iz)
  call ptospc(ng,ng,wkq,wka,xfactors,yfactors,xtrig,ytrig)
  call yderiv(ng,ng,hrky,wka,wkc)

  ! Apply de-aliasing filter and complete definition of A:
  aa(:,:,iz)=filt*(wkb+wkc)

  ! Need -(A + delta) in physical space for computing w just below:
  wka=aa(:,:,iz)+ds(:,:,iz)
  call spctop(ng,ng,wka,wkq,xfactors,yfactors,xtrig,ytrig)
  rsrc(:,:,iz)=-wkq
enddo

! Calculate vertical velocity (0 at iz = 0):
w(:,:,1)=dz2*(rsrc(:,:,0)+rsrc(:,:,1))
do iz=1,nz-1
  w(:,:,iz+1)=w(:,:,iz)+dz2*(rsrc(:,:,iz)+rsrc(:,:,iz+1))
enddo

! Complete definition of w by adding u*z_x + v*z_y after de-aliasing:
do iz=1,nz
  wkq=u(:,:,iz)*zx(:,:,iz)+v(:,:,iz)*zy(:,:,iz)
  call deal2d(wkq)
  w(:,:,iz)=w(:,:,iz)+wkq
enddo

return
end subroutine

!=======================================================================

subroutine source(sqs,sds,sgs)

! Gets the nonlinear source terms for linearised PV, divergence and 
! acceleration divergence  --- all in spectral space.  These are 
! returned in sqs, sds and sgs respectively.

! Note that (sds,sgs) only include the nonlinear terms for a 
! semi-implicit treatment, closely analogous to that described in 
! the appendix of Mohebalhojeh & Dritschel (2004).

! The spectral fields qs, ds and gs are all spectrally truncated.
! Note: u, v & zeta obtained by main_invert, and z obtained by psolve
! (which calls vertical) before calling this routine are all 
! spectrally truncated.

implicit none

 !Passed variables:
double precision:: sqs(ng,ng,0:nz),sds(ng,ng,0:nz),sgs(ng,ng,0:nz)

 !Local variables (physical):
double precision:: dd(ng,ng),ff(ng,ng),wkp(ng,ng),wkq(ng,ng)

 !Local variables (spectral):
double precision:: wka(ng,ng),wkb(ng,ng),wkc(ng,ng),wkd(ng,ng)

integer:: iz

!---------------------------------------------------------------
 !Calculate vertically-independent part of gs source (wkd):
wkd=zero
do iz=0,nz
  wkd=wkd+weight(iz)*aa(:,:,iz)
enddo
 !Note: aa contains div(u*rho_theta) in spectral space
wkd=c2g2*wkd

 !Loop over layers:
do iz=0,nz

   !qs source:

   !Compute div(ql*u,ql*v) (wka in spectral space):
  wka=qs(:,:,iz)
  call spctop(ng,ng,wka,wkq,xfactors,yfactors,xtrig,ytrig)
   !wkq contains the linearised PV in physical space
  wkp=wkq*u(:,:,iz)
  wkq=wkq*v(:,:,iz)
   !Compute spectral divergence from physical fields:
  call divs(wkp,wkq,wka)

   !Compute Jacobian of F = (1/rho_theta)*dP'/dtheta & z (wkb, spectral):
  ff=ri(:,:,iz)*dpn(:,:,iz)
  call deal2d(ff)
  wkq=z(:,:,iz)
  call jacob(ff,wkq,wkb)

   !Sum to get qs source:
  sqs(:,:,iz)=filt*(wkb-wka)

!---------------------------------------------------------------
 !Nonlinear part of ds source:

   !Compute J(u,v) (wkc in spectral space):
  call jacob(u(:,:,iz),v(:,:,iz),wkc)

   !Convert ds to physical space as dd:
  wka=ds(:,:,iz)
  call spctop(ng,ng,wka,dd,xfactors,yfactors,xtrig,ytrig)

   !Compute div(F*grad{z}-delta*{u,v}) (wkb in spectral space):
  wkp=ff*zx(:,:,iz)-dd*u(:,:,iz)
  wkq=ff*zy(:,:,iz)-dd*v(:,:,iz)
  call divs(wkp,wkq,wkb)

   !Add Lap(P') and complete definition of ds source:
  sds(:,:,iz)=filt*(two*wkc+wkb-hlap*ps(:,:,iz))

!---------------------------------------------------------------
   !Nonlinear part of gs source:
  sgs(:,:,iz)=cof*sqs(:,:,iz)+wkd-fsq*aa(:,:,iz)

enddo

return
end subroutine

!=======================================================================

subroutine diagnose

! Computes various quantities every time step to monitor the flow evolution.

implicit none

 !Local variables:
double precision:: umax,zrms,zmax
double precision:: uio,vio

!----------------------------------------------------------------------
 !Compute maximum horizontal speed:
umax=sqrt(maxval(u**2+v**2))

 !Compute maximum vertical vorticity:
zmax=maxval(abs(zeta))

 !Compute rms vertical vorticity:
zrms=sqrt(vsumi*(f12*sum(zeta(:,:,0)**2+zeta(:,:,nz)**2)+ &
                     sum(zeta(:,:,1:nzm1)**2)))

 !Record diagnostics to monitor.asc:
write(17,'(1x,f12.5,4(1x,f12.6))') t,f12*zrms**2,zrms,zmax,umax

return
end subroutine

!=======================================================================

subroutine savegrid(igrids)

! Saves various 3D and 2D fields, as well as vertically-integrated 
! spectra close to the desired save time.

implicit none

 !Passed variable (record for direct access write):
integer:: igrids

 !Local variables:
real:: v3d(ng,ng,0:nz) !Physical (for output of 3D fields):
double precision:: wkp(ng,ng),wkq(ng,ng) !Physical
double precision:: wks(ng,ng)            !Spectral
double precision:: zspec(0:ng),dspec(0:ng),gspec(0:ng),tmpspec(0:ng)
double precision:: ekin,epot,etot
integer:: iz,k

!---------------------------------------------------------------
 !Compute kinetic energy:
wkp=(one+r(:,:,0))*(u(:,:,0)**2+v(:,:,0)**2)
ekin=f12*sum(wkp)
wkp=(one+r(:,:,nz))*(u(:,:,nz)**2+v(:,:,nz)**2+w(:,:,nz)**2)
ekin=ekin+f12*sum(wkp)
do iz=1,nzm1
  wkp=(one+r(:,:,iz))*(u(:,:,iz)**2+v(:,:,iz)**2+w(:,:,iz)**2)
  ekin=ekin+sum(wkp)
enddo
ekin=f12*gvol*hinv*ekin

 !Compute potential energy (same as SW expression):
wkp=(hinv*z(:,:,nz)-one)**2
epot=f12*garea*csq*sum(wkp)

 !Compute total energy:
etot=ekin+epot

 !Write energies to ecomp.asc:
write(16,'(f13.6,5(1x,f16.9))') t,zero,ekin,ekin,epot,etot
write(*,'(a,f13.6,a,f13.6)') ' t = ',t,'  E_tot = ',etot

!---------------------------------------------------------------
 !Compute vertically-averaged 1d vorticity, divergence and 
 !acceleration divergence spectra:
zspec=zero
dspec=zero
gspec=zero
do iz=0,nz
  wkp=zeta(:,:,iz)
  call ptospc(ng,ng,wkp,wks,xfactors,yfactors,xtrig,ytrig)
  call spec1d(wks,tmpspec)
  zspec=zspec+weight(iz)*tmpspec
  call spec1d(ds(:,:,iz),tmpspec)
  dspec=dspec+weight(iz)*tmpspec
  call spec1d(gs(:,:,iz),tmpspec)
  gspec=gspec+weight(iz)*tmpspec
enddo
 !Normalise to take into account uneven sampling of wavenumbers 
 !in each shell [k-1/2,k+1/2]:
zspec=spmf*zspec
dspec=spmf*dspec
gspec=spmf*gspec

write(51,'(f13.6,1x,i5)') t,kmaxred
 !kmaxred = kmax/sqrt(2) to avoid shells in the upper corner of the
 !          kx,ky plane which are not fully populated
do k=1,kmaxred
  write(51,'(4(1x,f12.8))') alk(k),log10(zspec(k)),log10(dspec(k)+1.d-32), &
                                                   log10(gspec(k)+1.d-32)
enddo
 !Note: alk(k) = log_10(k)

!---------------------------------------------------------------
 !Write various 3D gridded fields to direct access files:

 !PV field:
do iz=0,nz
  wks=qs(:,:,iz)
  call spctop(ng,ng,wks,wkp,xfactors,yfactors,xtrig,ytrig)
  v3d(:,:,iz)=real(wkp)
enddo
write(31,rec=igrids) real(t),v3d

 !Divergence field:
do iz=0,nz
  wks=ds(:,:,iz)
  call spctop(ng,ng,wks,wkp,xfactors,yfactors,xtrig,ytrig)
  v3d(:,:,iz)=real(wkp)
enddo
write(32,rec=igrids) real(t),v3d

 !Acceleration divergence field:
do iz=0,nz
  wks=gs(:,:,iz)
  call spctop(ng,ng,wks,wkp,xfactors,yfactors,xtrig,ytrig)
  v3d(:,:,iz)=real(wkp)
enddo
write(33,rec=igrids) real(t),v3d

 !Dimensionless thickness anomaly:
write(34,rec=igrids) real(t),real(r)

 !Vertical velocity:
write(35,rec=igrids) real(t),real(w)

 !Non-hydrostatic pressure:
write(36,rec=igrids) real(t),real(pn)

!---------------------------------------------------------------
 !Write various vertically-integrated 2D fields to direct access files:

 !Divergence:
wkp=-w(:,:,nz)/z(:,:,nz)
write(42,rec=igrids) real(t),real(wkp)

 !Relative vorticity:
wkp=zero
do iz=0,nz
  wkp=wkp+weight(iz)*zeta(:,:,iz)*(one+r(:,:,iz))
enddo
wkp=wkp*hbar/z(:,:,nz)
write(45,rec=igrids) real(t),real(wkp)

 !PV anomaly:
wkp=hbar*(wkp+cof)/z(:,:,nz)-cof
write(41,rec=igrids) real(t),real(wkp)

 !Acceleration divergence:
wkp=zero
do iz=0,nz
  wks=gs(:,:,iz)
  call spctop(ng,ng,wks,wkq,xfactors,yfactors,xtrig,ytrig)
  wkp=wkp+weight(iz)*wkq*(one+r(:,:,iz))
enddo
wkp=wkp*hbar/z(:,:,nz)
write(43,rec=igrids) real(t),real(wkp)

 !Dimensionless height anomaly:
wkp=hinv*z(:,:,nz)-one
write(44,rec=igrids) real(t),real(wkp)

return
end subroutine

!=======================================================================

subroutine finalise

implicit none

write(*,*) ' Code completed normally'

 !Close output files (opened in subroutine initialise):
close(16)
close(17)

close(31)
close(32)
close(33)
close(34)
close(35)
close(36)

close(41)
close(42)
close(43)
close(44)
close(45)

close(51)

return
end subroutine

 !End main program
end program nhswps

!=======================================================================
