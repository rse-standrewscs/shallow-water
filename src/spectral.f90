module spectral

! Module containing subroutines for spectral operations, inversion, etc.

use constants
use sta2dfft

 !Common arrays, constants:

 !Spectral operators:
double precision:: hlap(ng,ng),glap(ng,ng),rlap(ng,ng),helm(ng,ng)
double precision:: c2g2(ng,ng),simp(ng,ng),rope(ng,ng),fope(ng,ng)
double precision:: filt(ng,ng),diss(ng,ng),opak(ng,ng),rdis(ng,ng)

 !Tridiagonal arrays for the pressure Poisson equation:
double precision:: etdv(ng,ng,0:nz-1),htdv(ng,ng,0:nz-1),ap(ng,ng)

 !Tridiagonal arrays for the compact difference calculation of d/dz:
double precision:: etd1(nz),htd1(nz)

 !Array for theta and vertical weights for integration:
double precision:: theta(0:nz),weight(0:nz)

 !For 2D FFTs:
double precision:: hrkx(ng),hrky(ng),rk(ng)
double precision:: xtrig(2*ng),ytrig(2*ng)
integer:: xfactors(5),yfactors(5)

double precision:: spmf(0:ng),alk(ng)
integer:: kmag(ng,ng),kmax,kmaxred


!::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 !Internal subroutine definitions (inherit global variables):
!::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

contains

!=============================================================

subroutine init_spectral
! Initialises this module

implicit none

!Local variables:
double precision:: a0(ng,ng),a0b(ng,ng),apb(ng,ng)
double precision:: rkmax,rks,snorm,rrsq
double precision:: anu,rkfsq
integer:: kx,ky,k,iz

!----------------------------------------------------------------------
 !Set up 2D FFTs:
call init2dfft(ng,ng,twopi,twopi,xfactors,yfactors,xtrig,ytrig,hrkx,hrky)

 !Define wavenumbers and filtered wavenumbers:
rk(1)=zero
do k=1,ng/2-1
  rk(k+1)   =hrkx(2*k)
  rk(ng+1-k)=hrkx(2*k)
enddo
rk(ng/2+1)=hrkx(ng)

!-----------------------------------------------------------------------
 !Initialise arrays for computing the spectrum of any field:
rkmax=dble(ng/2)
kmax=nint(rkmax*sqrt(two))
do k=0,kmax
  spmf(k)=zero
enddo
do ky=1,ng
  do kx=1,ng
    k=nint(sqrt(rk(kx)**2+rk(ky)**2))
    kmag(kx,ky)=k
    spmf(k)=spmf(k)+one
  enddo
enddo
 !Compute spectrum multiplication factor (spmf) to account for unevenly
 !sampled shells and normalise spectra by 8/(ng*ng) so that the sum
 !of the spectrum is equal to the L2 norm of the original field:
snorm=four*pi/dble(ng*ng)
spmf(0)=zero
do k=1,kmax
  spmf(k)=snorm*dble(k)/spmf(k)
  alk(k)=log10(dble(k))
enddo
 !Only output shells which are fully occupied (k <= kmaxred):
kmaxred=ng/2

!-----------------------------------------------------------------------
 !Define a variety of spectral operators:

 !Hyperviscosity coefficient (Dritschel, Gottwald & Oliver, JFM (2017)):
anu=cdamp*cof/rkmax**(2*nnu)
 !Assumes Burger number = 1.

 !Used for de-aliasing filter below:
rkfsq=(dble(ng)/3.d0)**2

do ky=1,ng
  do kx=1,ng
    rks=rk(kx)**2+rk(ky)**2
     !grad^2:
    hlap(kx,ky)=-rks
     !Spectral c^2*grad^2 - f^2 operator (G in paper):
    opak(kx,ky)=-(fsq+csq*rks)
     !Hyperviscous operator:
    diss(kx,ky)=anu*rks**nnu
     !De-aliasing filter:
    if (rks .gt. rkfsq) then
      filt(kx,ky)=zero
      glap(kx,ky)=zero
      c2g2(kx,ky)=zero
      rlap(kx,ky)=zero
      helm(kx,ky)=zero
      rope(kx,ky)=zero
      rdis(kx,ky)=zero
    else
      filt(kx,ky)=one
       !-g*grad^2:
      glap(kx,ky)=gravity*rks
       !c^2*grad^2:
      c2g2(kx,ky)=-csq*rks
       !grad^{-2} (inverse Laplacian):
      rlap(kx,ky)=-one/(rks+1.d-20)
       !(c^2*grad^2 - f^2)^{-1} (G^{-1} in paper):
      helm(kx,ky)=one/opak(kx,ky)
       !c^2*grad^2/(c^2*grad^2 - f^2) (used in layer thickness inversion):
      rope(kx,ky)=c2g2(kx,ky)*helm(kx,ky)
      rdis(kx,ky)=dt2i+diss(kx,ky)
    endif
     !Operators needed for semi-implicit time stepping:
    rrsq=(dt2i+diss(kx,ky))**2
    fope(kx,ky)=-c2g2(kx,ky)/(rrsq-opak(kx,ky))
     !Semi-implicit operator for inverting divergence:
    simp(kx,ky)=one/(rrsq+fsq)
     !Re-define damping operator for use in qd evolution:
    diss(kx,ky)=two/(one+dt2*diss(kx,ky))
  enddo
enddo

 !Ensure area averages remain zero:
rlap(1,1)=zero

!-----------------------------------------------------------------------
 !Define theta and the vertical weight for trapezoidal integration:
do iz=0,nz
  theta(iz)=dz*dble(iz)
  weight(iz)=one
enddo
weight(0)=f12
weight(nz)=f12
weight=weight/dble(nz)

!-----------------------------------------------------------------------
 !Tridiagonal coefficients depending only on kx and ky:
do ky=1,ng
  do kx=1,ng
    rks=rk(kx)**2+rk(ky)**2
    a0(kx,ky)=-two*dzisq-f56*rks
    a0b(kx,ky)=-dzisq-f13*rks
    ap(kx,ky)=dzisq-f112*rks
    apb(kx,ky)=dzisq-f16*rks
  enddo
enddo

!-----------------------------------------------------------------------
 !Tridiagonal arrays for the pressure:
htdv(:,:,0)=filt/a0b
etdv(:,:,0)=-apb*htdv(:,:,0)
do iz=1,nz-2
  htdv(:,:,iz)=filt/(a0+ap*etdv(:,:,iz-1))
  etdv(:,:,iz)=-ap*htdv(:,:,iz)
enddo
htdv(:,:,nz-1)=filt/(a0+ap*etdv(:,:,nz-2))

 !Tridiagonal arrays for the compact difference calculation of d/dz:
htd1(1)=one/f23
etd1(1)=-f16*htd1(1)
do iz=2,nz-1
  htd1(iz)=one/(f23+f16*etd1(iz-1))
  etd1(iz)=-f16*htd1(iz)
enddo
htd1(nz)=one/(f23+f13*etd1(nz-1))

return 
end subroutine

!======================================================================

subroutine main_invert(qs,ds,gs,r,u,v,zeta)
! Given the PV anomaly qs, divergence ds and acceleration divergence gs
! (all in spectral space), this routine computes the dimensionless
! layer thickness anomaly and horizontal velocity, as well as the 
! relative vertical vorticity in physical space.

implicit none

 !Passed variables:
double precision:: qs(ng,ng,0:nz),ds(ng,ng,0:nz),gs(ng,ng,0:nz)
double precision:: u(ng,ng,0:nz),v(ng,ng,0:nz)
double precision:: r(ng,ng,0:nz),zeta(ng,ng,0:nz)

 !Local variables:
double precision:: es(ng,ng,0:nz)
double precision:: wka(ng,ng),wkb(ng,ng),wkc(ng,ng),wkd(ng,ng)
double precision:: wke(ng,ng),wkf(ng,ng),wkg(ng,ng),wkh(ng,ng)
double precision:: uio,vio
integer:: iz

!--------------------------------------------------------------
 !Define eta = gamma_l/f^2 - q_l/f (spectral):
es=cofi*(cofi*gs-qs)

 !Compute vertical average of eta (store in wkh):
wkh=zero
do iz=0,nz
  wkh=wkh+weight(iz)*es(:,:,iz)
enddo
 !Multiply by F = c^2*k^2/(f^2+c^2k^2) in spectral space:
wkh=rope*wkh

 !Initialise mean flow:
uio=zero
vio=zero

 !Complete inversion:
do iz=0,nz
   !Obtain layer thickness anomaly (spectral, in wka):
  wka=es(:,:,iz)-wkh

   !Obtain relative vorticity (spectral, in wkb):
  wkb=qs(:,:,iz)+cof*wka

   !Invert Laplace operator on zeta & delta to define velocity:
  wkc=rlap*wkb
  wkd=rlap*ds(:,:,iz)

   !Calculate derivatives spectrally:
  call xderiv(ng,ng,hrkx,wkd,wke)
  call yderiv(ng,ng,hrky,wkd,wkf)
  call xderiv(ng,ng,hrkx,wkc,wkd)
  call yderiv(ng,ng,hrky,wkc,wkg)

   !Define velocity components:
  wke=wke-wkg
  wkf=wkf+wkd

   !Bring quantities back to physical space and store:
  call spctop(ng,ng,wka,wkc,xfactors,yfactors,xtrig,ytrig)
  r(:,:,iz)=wkc
  call spctop(ng,ng,wkb,wkd,xfactors,yfactors,xtrig,ytrig)
  zeta(:,:,iz)=wkd
  call spctop(ng,ng,wke,wka,xfactors,yfactors,xtrig,ytrig)
  u(:,:,iz)=wka
  call spctop(ng,ng,wkf,wkb,xfactors,yfactors,xtrig,ytrig)
  v(:,:,iz)=wkb

   !Accumulate mean flow (uio,vio):
  uio=uio-weight(iz)*sum(wkc*wka)*dsumi
  vio=vio-weight(iz)*sum(wkc*wkb)*dsumi
enddo

 !Add mean flow:
u=u+uio
v=v+vio

return
end subroutine

!=================================================================

subroutine jacob(aa,bb,cs)
! Computes the (xy) Jacobian of aa and bb and returns it in cs.
! aa and bb are in physical space while cs is in spectral space

! NOTE: aa and bb are assumed to be spectrally truncated (de-aliased).

implicit none

 !Passed arrays:
double precision:: aa(ng,ng),bb(ng,ng),cs(ng,ng)

 !Work arrays:
double precision:: ax(ng,ng),ay(ng,ng),bx(ng,ng),by(ng,ng)
double precision:: wka(ng,ng),wkb(ng,ng)

!---------------------------------------------------------
wkb=aa
call ptospc(ng,ng,wkb,wka,xfactors,yfactors,xtrig,ytrig)
 !Get derivatives of aa:
call xderiv(ng,ng,hrkx,wka,wkb)
call spctop(ng,ng,wkb,ax,xfactors,yfactors,xtrig,ytrig)
call yderiv(ng,ng,hrky,wka,wkb)
call spctop(ng,ng,wkb,ay,xfactors,yfactors,xtrig,ytrig)

wkb=bb
call ptospc(ng,ng,wkb,wka,xfactors,yfactors,xtrig,ytrig)
 !Get derivatives of bb:
call xderiv(ng,ng,hrkx,wka,wkb)
call spctop(ng,ng,wkb,bx,xfactors,yfactors,xtrig,ytrig)
call yderiv(ng,ng,hrky,wka,wkb)
call spctop(ng,ng,wkb,by,xfactors,yfactors,xtrig,ytrig)

wkb=ax*by-ay*bx
call ptospc(ng,ng,wkb,cs,xfactors,yfactors,xtrig,ytrig)
 !The output is *not* spectrally truncated!

return
end subroutine

!=================================================================

subroutine divs(aa,bb,cs)
! Computes the divergence of (aa,bb) and returns it in cs.
! Both aa and bb in physical space but cs is in spectral space.

implicit none

 !Passed arrays:
double precision:: aa(ng,ng),bb(ng,ng)   !Physical
double precision:: cs(ng,ng)             !Spectral

 !Work arrays:
double precision:: wkp(ng,ng)            !Physical
double precision:: wka(ng,ng),wkb(ng,ng) !Spectral

!---------------------------------------------------------
wkp=aa
call ptospc(ng,ng,wkp,wka,xfactors,yfactors,xtrig,ytrig)
call xderiv(ng,ng,hrkx,wka,wkb)

wkp=bb
call ptospc(ng,ng,wkp,wka,xfactors,yfactors,xtrig,ytrig)
call yderiv(ng,ng,hrky,wka,cs)

cs=wkb+cs

return
end subroutine

!=================================================================

subroutine ptospc3d(fp,fs,izbeg,izend)
! Transforms a physical 3d field fp to spectral space (horizontally)
! as the array fs.

implicit none

 !Passed variables:
double precision:: fp(ng,ng,0:nz)  !Physical
double precision:: fs(ng,ng,0:nz)  !Spectral
integer:: izbeg,izend

 !Work arrays:
double precision:: wkp(ng,ng)  !Physical
double precision:: wks(ng,ng)  !Spectral
integer:: iz

!---------------------------------------------------------
do iz=izbeg,izend
  wkp=fp(:,:,iz)
  call ptospc(ng,ng,wkp,wks,xfactors,yfactors,xtrig,ytrig)
  fs(:,:,iz)=wks
enddo

return
end subroutine

!=================================================================

subroutine spctop3d(fs,fp,izbeg,izend)
! Transforms a spectral 3d field fs to physical space (horizontally)
! as the array fp.

implicit none

 !Passed variables:
double precision:: fp(ng,ng,0:nz)  !Physical
double precision:: fs(ng,ng,0:nz)  !Spectral
integer:: izbeg,izend

 !Work arrays:
double precision:: wkp(ng,ng)  !Physical
double precision:: wks(ng,ng)  !Spectral
integer:: iz

!---------------------------------------------------------
do iz=izbeg,izend
  wks=fs(:,:,iz)
  call spctop(ng,ng,wks,wkp,xfactors,yfactors,xtrig,ytrig)
  fp(:,:,iz)=wkp
enddo

return
end subroutine

!=================================================================

subroutine deal3d(fp)
! Filters (horizontally) a physical 3d field fp (overwrites fp).

implicit none

 !Passed variable:
double precision:: fp(ng,ng,0:nz)  !Physical

 !Local variables:
double precision:: wkp(ng,ng)  !Physical
double precision:: wks(ng,ng)  !Spectral
integer:: iz

do iz=0,nz
  wkp=fp(:,:,iz)
  call ptospc(ng,ng,wkp,wks,xfactors,yfactors,xtrig,ytrig)
  wks=filt*wks
  call spctop(ng,ng,wks,wkp,xfactors,yfactors,xtrig,ytrig)
  fp(:,:,iz)=wkp
enddo

return
end subroutine

!=================================================================

subroutine deal2d(fp)
! Filters (horizontally) a physical 2d field fp (overwrites fp).

implicit none

 !Passed variable:
double precision:: fp(ng,ng)  !Physical

 !Local variable:
double precision:: fs(ng,ng)  !Spectral

call ptospc(ng,ng,fp,fs,xfactors,yfactors,xtrig,ytrig)
fs=filt*fs
call spctop(ng,ng,fs,fp,xfactors,yfactors,xtrig,ytrig)

return
end subroutine

!===================================================================

subroutine spec1d(ss,spec)
! Computes the 1d spectrum of a spectral field ss and returns the
! result in spec.

implicit none

 !Passed variables:
double precision:: ss(ng,ng),spec(0:ng)

 !Local variables:
integer:: kx,ky,k

!--------------------------------------------------------
do k=0,kmax
  spec(k)=zero
enddo

 !x and y-independent mode:
k=kmag(1,1)
spec(k)=spec(k)+f14*ss(1,1)**2

 !y-independent mode:
do kx=2,ng
  k=kmag(kx,1)
  spec(k)=spec(k)+f12*ss(kx,1)**2
enddo

 !x-independent mode:
do ky=2,ng
  k=kmag(1,ky)
  spec(k)=spec(k)+f12*ss(1,ky)**2
enddo

 !All other modes:
do ky=2,ng
  do kx=2,ng
    k=kmag(kx,ky)
    spec(k)=spec(k)+ss(kx,ky)**2
  enddo
enddo

return
end subroutine

!===================================================================

end module     
