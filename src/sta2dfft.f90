module sta2dfft

use stafft

! This module performs FFTs in two directions on two dimensional arrays using 
! the stafft library module to actually compute the FFTs. If FFTs in one 
! direction only are required use the stafft module directly. The module can 
! compute any combination of sine, cosine and full FFTs in each direction. 
! Along with the usual forwards (physical -> Fourier space) and reverse 
! (Fourier space -> physical) routines there are also routines for computing 
! the first derivatives in either direction.
!
! The convention is that for each direction the array is dimensioned 1:nx or 
! 1:ny for either the sine or full transforms. While the cosine transforms 
! require the additional endpoint so 0:nx or 0:ny.
!
! The routines contained in this module are:
!
! init2dfft(nx,ny,lx,ly,xfactors,yfactors,xtrig,ytrig,kx,ky)
!          This routine initialises all the arrays needed for further 
!          transforms. The integers nx and ny are the array dimensions. Then 
!          lx and ly are the domain lengths - these are needed for the correct 
!          scaling when computing derivatives. The arrays xfactors, yfactors, 
!          xtrig and ytrig are needed to perform the various FFTs by the stafft 
!          module (see there for further details. kx and ky are arrays to hold 
!          the wavenumbers associated with each mode in the domain, and are 
!          used in computing derivatives. 
!
!          **If it is known at initialisation that no derivatives are required 
!            it is possible just to pass 1.d0 for each of lx and ly, along with
!            dummy arrays for kx and ky since these are only needed for 
!            computing the derviatives.**

 
implicit none

contains

!=====================================================================
subroutine init2dfft(nx,ny,lx,ly,xfactors,yfactors,xtrig,ytrig,kx,ky)

! This subroutine performs the initialisation work for all subsequent
! transform and derivative routines. 
! It calls the initfft() routine from the supproting 1d FFT module for 
! transforms in both x and y directions.
! The routine then defines the two wavenumber arrays, one in each direction.

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: lx,ly,kx(nx),ky(ny)

 !Local declarations:
double precision,parameter:: pi=3.141592653589793238462643383279502884197169399375105820974944592307816d0
double precision:: sc
integer:: k

!----------------------------------------------
call initfft(nx,xfactors,xtrig)
call initfft(ny,yfactors,ytrig)
 
if (lx .ne. 0.d0) then
   !Define x wavenumbers:
  sc=pi/lx
  do k=1,nx
    kx(k)=sc*dble(k)
  enddo
else
   !Catastrophic end to run if wave number definition fails:
  write(*,*) '**********************************************'
  write(*,*) ' Wavenumber array definition not possible.'
  write(*,*) ' Domain length in x equal to zero not allowed.'
  write(*,*) ' STOPPING...'
  write(*,*) '**********************************************'
  stop
endif

if (ly .ne. 0.d0) then
   !Define y wavenumbers:
  sc=pi/ly
  do k=1,ny
    ky(k)=sc*dble(k)
  enddo
else
   !Catastrophic end to run if wave number definition fails:
  write(*,*) '**********************************************'
  write(*,*) ' Wavenumber array definition not possible.'
  write(*,*) ' Domain length in y equal to zero not allowed.'
  write(*,*) ' STOPPING...'
  write(*,*) '**********************************************'
  stop
endif

return
end subroutine

!===================================================
! These routines are for arrays periodic in x and y.
!===================================================

subroutine ptospc(nx,ny,rvar,svar,xfactors,yfactors,xtrig,ytrig)
! Performs a physical -> spectral transform of a variable 
! rvar(ny,nx) periodic in x and y, and returns the result 
! (transposed) in svar(nx,ny).
! *** Note rvar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(ny,nx),svar(nx,ny)
 !Local declarations:
integer:: kx,iy

 !Carry out a full x transform first:
call forfft(ny,nx,rvar,xtrig,xfactors) 

 !Transpose array:
do kx=1,nx
  do iy=1,ny
    svar(kx,iy)=rvar(iy,kx)
  enddo
enddo

 !Carry out a full y transform on transposed array:
call forfft(nx,ny,svar,ytrig,yfactors)

return
end subroutine

!===================================================================

subroutine spctop(nx,ny,svar,rvar,xfactors,yfactors,xtrig,ytrig)
! Performs a spectral -> physical transform of a variable 
! svar(nx,ny) periodic in x and y and returns the result 
! (transposed) in rvar(ny,nx).
! *** Note svar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(ny,nx),svar(nx,ny)
 !Local declarations:
integer:: kx,iy

 !Carry out a full inverse y transform first:
call revfft(nx,ny,svar,ytrig,yfactors)

 !Transpose array:
do kx=1,nx
  do iy=1,ny
    rvar(iy,kx)=svar(kx,iy)
  enddo
enddo

 !Carry out a full inverse x transform:
call revfft(ny,nx,rvar,xtrig,xfactors) 

return
end subroutine

!===================================================================

subroutine xderiv(nx,ny,rkx,var,der)
! Computes der = d(var)/dx, spectrally, for a variable 
! var(nx,ny) periodic in x and y.
! *** both var and der are spectral ***

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rkx(nx),var(nx,ny),der(nx,ny)
 !Local declarations:
integer:: nwx,nxp2,kx,ky,dkx,kxc

nwx=nx/2
nxp2=nx+2
 !Carry out differentiation by wavenumber multiplication:
do ky=1,ny
  der(1,ky)=0.d0
  do kx=2,nx-nwx
    dkx=2*(kx-1)
    kxc=nxp2-kx
    der(kx, ky)=-rkx(dkx)*var(kxc,ky)
    der(kxc,ky)= rkx(dkx)*var(kx ,ky)
  enddo
enddo

if (mod(nx,2) .eq. 0) then
  kxc=nwx+1
  do ky=1,ny
    der(kxc,ky)=0.d0
  enddo
endif

return
end subroutine

!===================================================================

subroutine yderiv(nx,ny,rky,var,der)
! Computes der = d(var)/dy, spectrally, for a variable 
! var(nx,ny) periodic in x and y.
! *** both var and der are spectral ***

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rky(ny),var(nx,ny),der(nx,ny)
 !Local declarations:
integer:: nwy,nyp2,kx,ky,kyc
double precision:: fac 

nwy=ny/2
nyp2=ny+2
 !Carry out differentiation by wavenumber multiplication:
do kx=1,nx
  der(kx,1)=0.d0
enddo

do ky=2,ny-nwy
  kyc=nyp2-ky
  fac=rky(2*(ky-1))
  do kx=1,nx
    der(kx, ky)=-fac*var(kx,kyc)
    der(kx,kyc)= fac*var(kx ,ky)
  enddo
enddo

if (mod(ny,2) .eq. 0) then
  kyc=nwy+1
  do kx=1,nx
    der(kx,kyc)=0.d0
  enddo
endif

return
end subroutine

!===================================================
! These routines are for arrays periodic in x only.
!===================================================

subroutine ptospc_fc(nx,ny,rvar,svar,xfactors,yfactors,xtrig,ytrig)
! Performs a physical -> spectral transform of a variable 
! rvar(0:ny,nx) periodic in x and represented by a cosine 
! series in y and returns the result (transposed) in 
! svar(nx,0:ny).
! *** Note rvar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(0:ny,nx),svar(nx,0:ny)
 !Local declarations:
integer:: kx,iy

 !Carry out a full x transform first:
call forfft(ny+1,nx,rvar,xtrig,xfactors) 

 !Transpose array:
do kx=1,nx
  do iy=0,ny
    svar(kx,iy)=rvar(iy,kx)
  enddo
enddo

 !Carry out y cosine transform on transposed array:
call dct(nx,ny,svar,ytrig,yfactors)

return
end subroutine

!===================================================================

subroutine ptospc_fs(nx,ny,rvar,svar,xfactors,yfactors,xtrig,ytrig)
! Performs a physical -> spectral transform of a variable 
! rvar(ny,nx) periodic in x and represented by a sine series in y
! and returns the result (transposed) in svar(nx,ny). 
! *** Note rvar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(ny,nx),svar(nx,ny)
 !Local declarations:
integer:: kx,iy
 
 !Carry out a full x transform first:
call forfft(ny,nx,rvar,xtrig,xfactors) 

 !Transpose array:
do kx=1,nx
  do iy=1,ny-1
    svar(kx,iy)=rvar(iy,kx)
  enddo
  svar(kx,ny)=0.d0
enddo

 !Carry out y sine transform on transposed array:
call dst(nx,ny,svar,ytrig,yfactors)

return
end subroutine

!===================================================================

subroutine spctop_fc(nx,ny,svar,rvar,xfactors,yfactors,xtrig,ytrig)
! Performs a spectral -> physical transform of a variable 
! svar(nx,0:ny) periodic in x and represented by a cosine 
! series in y and returns the result (transposed) in 
! rvar(0:ny,nx)
! *** Note svar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(0:ny,nx),svar(nx,0:ny)
 !Local declarations:
integer:: kx,iy

 !Carry out y cosine transform first:
call dct(nx,ny,svar,ytrig,yfactors)

 !Transpose array:
do kx=1,nx
  do iy=0,ny
    rvar(iy,kx)=svar(kx,iy)
  enddo
enddo

 !Carry out a full inverse x transform:
call revfft(ny+1,nx,rvar,xtrig,xfactors) 

return
end subroutine

!===================================================================

subroutine spctop_fs(nx,ny,svar,rvar,xfactors,yfactors,xtrig,ytrig)
! Performs a spectral -> physical transform of a variable 
! svar(nx,ny) periodic in x and represented by a sine series 
! in y and returns the result (transposed) in rvar(ny,nx).
! *** Note svar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(ny,0:nx-1),svar(0:nx-1,ny)
 !Local declarations:
integer:: kx,iy

 !Carry out y sine transform first:
call dst(nx,ny,svar,ytrig,yfactors)

 !Transpose array:
do kx=0,nx-1
  do iy=1,ny-1
    rvar(iy,kx)=svar(kx,iy)
  enddo
  rvar(ny,kx)=0.d0
enddo

 !Carry out a full inverse x transform:
call revfft(ny,nx,rvar,xtrig,xfactors) 

return
end subroutine

!===================================================================

subroutine xderiv_fs(nx,ny,rkx,var,der)
! Computes der = d(var)/dx, spectrally, for a variable 
! var(nx,ny) periodic in x and represented by a sine 
! series in y.
! *** both var and der are spectral ***

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rkx(nx),var(nx,ny),der(nx,ny)
 !Local declarations:
integer:: nwx,nxp2,kx,ky,dkx,kxc

nwx=nx/2
nxp2=nx+2
 !Carry out differentiation by wavenumber multiplication:
do ky=1,ny-1
  der(1,ky)=0.d0
  do kx=2,nx-nwx
    dkx=2*(kx-1)
    kxc=nxp2-kx
    der(kx, ky)=-rkx(dkx)*var(kxc,ky)
    der(kxc,ky)= rkx(dkx)*var(kx ,ky)
  enddo
enddo

if (mod(nx,2) .eq. 0) then
  kxc=nwx+1
  do ky=1,ny-1
    der(kxc,ky)=0.d0
  enddo
endif

return
end subroutine

!===================================================================

subroutine xderiv_fc(nx,ny,rkx,var,der)
! Computes der = d(var)/dx, spectrally, for a variable 
! var(nx,0:ny) periodic in x and represented by a cosine
! series in y.
! *** both var and der are spectral ***

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rkx(nx),var(nx,0:ny),der(nx,0:ny)
 !Local declarations:
integer:: nwx,nxp2,kx,ky,dkx,kxc

nwx=nx/2
nxp2=nx+2
 !Carry out differentiation by wavenumber multiplication:
do ky=0,ny
  der(1,ky)=0.d0
  do kx=2,nx-nwx
    dkx=2*(kx-1)
    kxc=nxp2-kx
    der(kx, ky)=-rkx(dkx)*var(kxc,ky)
    der(kxc,ky)= rkx(dkx)*var(kx ,ky)
  enddo
enddo

if (mod(nx,2) .eq. 0) then
  kxc=nwx+1
  do ky=0,ny
    der(kxc,ky)=0.d0
  enddo
endif

return
end subroutine

!===================================================================

subroutine yderiv_fs(nx,ny,rky,var,der)
! Computes der = d(var)/dy, spectrally, for a variable 
! var(nx,ny) periodic in x and represented by a sine 
! series in y.
! *** both var and der are spectral ***
! ==> der(nx,0:ny) changes to a cosine series in y

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rky(ny),var(nx,ny),der(nx,0:ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do kx=1,nx
  der(kx,0)=0.d0
   !The above implies the mean value of der for every kx is 0.d0.
  der(kx,ny)=0.d0
   !der = 0 when ky = ny since var = 0 when ky = ny.
enddo
do ky=1,ny-1
  do kx=1,nx
    der(kx,ky)=rky(ky)*var(kx,ky)
  enddo
enddo

return
end subroutine

!===================================================================

subroutine yderiv_fc(nx,ny,rky,var,der)
! Computes der = d(var)/dy, spectrally, for a variable 
! var(nx,0:ny) periodic in x and represented by a cosine 
! series in y.
! *** both var and der are spectral ***
! ==> der(nx,ny) changes to a sine series in y

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rky(ny),var(nx,0:ny),der(nx,ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do kx=1,nx
  der(kx,ny)=0.d0
enddo
do ky=1,ny
  do kx=1,nx
    der(kx,ky)=-rky(ky)*var(kx,ky)
  enddo
enddo

return
end subroutine

!===================================================================
! These routines are for arrays periodic in neither x nor y.
!===================================================================

subroutine ptospc_cc(nx,ny,rvar,svar,xfactors,yfactors,xtrig,ytrig)
! Performs a physical -> spectral transform of a variable rvar(0:ny,0:nx)
! represented by a cosine series in x and a cosine series in y
! and returns the result (transposed) in svar(0:nx,0:ny).
! *** Note rvar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(0:ny,0:nx),svar(0:nx,0:ny)
 !Local declarations:
integer:: kx,iy

 !Carry out x cosine transform first:
call dct(ny+1,nx,rvar,xtrig,xfactors)

 !Transpose array:
do kx=0,nx
  do iy=0,ny
    svar(kx,iy)=rvar(iy,kx)
  enddo
enddo

 !Carry out y cosine transform on transposed array:
call dct(nx+1,ny,svar,ytrig,yfactors)

return
end subroutine

!===================================================================

subroutine ptospc_cs(nx,ny,rvar,svar,xfactors,yfactors,xtrig,ytrig)
! Performs a physical -> spectral transform of a variable rvar(ny,0:nx)
! represented by a cosine series in x and a  sine  series in y
! and returns the result (transposed) in svar(0:nx,ny)
! *** Note rvar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(ny,0:nx),svar(0:nx,ny)
 !Local declarations:
integer:: kx,iy

 !Carry out x cosine transform first:
call dct(ny,nx,rvar,xtrig,xfactors)

 !Transpose array:
do kx=0,nx
  do iy=1,ny-1
    svar(kx,iy)=rvar(iy,kx)
  enddo
  svar(kx,ny)=0.d0
enddo

 !Carry out y sine transform on transposed array:
call dst(nx+1,ny,svar,ytrig,yfactors)

return
end subroutine

!===================================================================

subroutine ptospc_sc(nx,ny,rvar,svar,xfactors,yfactors,xtrig,ytrig)
! Performs a physical -> spectral transform of a variable rvar(0:ny,nx)
! represented by a  sine  series in x and a cosine series in y
! and returns the result (transposed) in svar(nx,0:ny)
! *** Note rvar is destroyed on return. ***

implicit none

 !Argument declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(0:ny,nx),svar(nx,0:ny)
 !Local declarations
integer:: kx,iy

 !Carry out x sine transform first:
call dst(ny+1,nx,rvar,xtrig,xfactors)

 !Transpose array:
do kx=1,nx-1
  do iy=0,ny
    svar(kx,iy)=rvar(iy,kx)
  enddo
enddo
do iy=0,ny
  svar(nx,iy)=0.d0
enddo

 !Carry out y cosine transform on transposed array:
call dct(nx,ny,svar,ytrig,yfactors) 

return
end subroutine

!===================================================================

subroutine ptospc_ss(nx,ny,rvar,svar,xfactors,yfactors,xtrig,ytrig)
! Performs a physical -> spectral transform of a variable rvar(ny,nx)
! represented by a  sine  series in x and a  sine  series in y
! and returns the result (transposed) in svar(nx,ny).
! *** Note rvar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(ny,nx),svar(nx,ny)
 !Local declarations:
integer:: kx,iy

 !Carry out x sine transform first:
call dst(ny,nx,rvar,xtrig,xfactors)

 !Transpose array:
do kx=1,nx-1
  do iy=1,ny-1
    svar(kx,iy)=rvar(iy,kx)
  enddo
  svar(kx,ny)=0.d0
enddo
do iy=1,ny
  svar(nx,iy)=0.d0
enddo

 !Carry out y sine transform on transposed array:
call dst(nx,ny,svar,ytrig,yfactors)

return
end subroutine

!===================================================================

subroutine spctop_cc(nx,ny,svar,rvar,xfactors,yfactors,xtrig,ytrig)
! Performs a spectral -> physical transform of a variable svar(0:nx,0:ny)
! represented by a cosine series in x and a cosine series in y
! and returns the result (transposed) in rvar(0:ny,0:nx)
! *** Note svar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(0:ny,0:nx),svar(0:nx,0:ny)
 !Local declarations:
integer:: kx,iy

 !Carry out y cosine transform first:
call dct(nx+1,ny,svar,ytrig,yfactors)

 !Transpose array:
do kx=0,nx
  do iy=0,ny
    rvar(iy,kx)=svar(kx,iy)
  enddo
enddo

 !Carry out x cosine transform on transposed array:
call dct(ny+1,nx,rvar,xtrig,xfactors)

return
end subroutine

!===================================================================

subroutine spctop_cs(nx,ny,svar,rvar,xfactors,yfactors,xtrig,ytrig)
! Performs a spectral -> physical transform of a variable svar(0:nx,ny)
! represented by a cosine series in x and a  sine  series in y
! and returns the result (transposed) in rvar(ny,0:nx)
! *** Note svar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(ny,0:nx),svar(0:nx,ny)
 !Local declarations:
integer:: kx,iy

 !Carry out y sine transform first:
call dst(nx+1,ny,svar,ytrig,yfactors)

 !Transpose array:
do kx=0,nx
  do iy=1,ny-1
    rvar(iy,kx)=svar(kx,iy)
  enddo
  rvar(ny,kx)=0.d0
enddo

 !Carry out x cosine transform on transposed array:
call dct(ny,nx,rvar,xtrig,xfactors)

return
end subroutine

!===================================================================

subroutine spctop_sc(nx,ny,svar,rvar,xfactors,yfactors,xtrig,ytrig)
! Performs a spectral -> physical transform of a variable svar(nx,0:ny)
! represented by a  sine  series in x and a cosine series in y
! and returns the result (transposed) in rvar(0:ny,nx)
! *** Note svar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(0:ny,nx),svar(nx,0:ny)
 !Local declarations:
integer:: kx,iy

 !Carry out y cosine transform first:
call dct(nx,ny,svar,ytrig,yfactors)

 !Transpose array:
do kx=1,nx-1
  do iy=0,ny
    rvar(iy,kx)=svar(kx,iy)
  enddo
enddo
do iy=0,ny
  rvar(iy,nx)=0.d0
enddo

 !Carry out x sine transform on transposed array:
call dst(ny+1,nx,rvar,xtrig,xfactors)

return
end subroutine

!===================================================================

subroutine spctop_ss(nx,ny,svar,rvar,xfactors,yfactors,xtrig,ytrig)
! Performs a spectral -> physical transform of a variable svar(nx,ny)
! represented by a  sine  series in x and a  sine  series in y
! and returns the result (transposed) in rvar(ny,nx)
! *** Note svar is destroyed on return. ***

implicit none

 !Arguments declarations:
integer:: nx,ny,xfactors(5),yfactors(5)
double precision:: xtrig(2*nx),ytrig(2*ny)
double precision:: rvar(ny,nx),svar(nx,ny)
 !Local declarations:
integer:: kx,iy

 !Carry out y sine transform first:
call dst(nx,ny,svar,ytrig,yfactors)

 !Transpose array:
do kx=1,nx-1
  do iy=1,ny-1
    rvar(iy,kx)=svar(kx,iy)
  enddo
  rvar(ny,kx)=0.d0
enddo
do iy=1,ny
  rvar(iy,nx)=0.d0
enddo

 !Carry out x sine transform on transposed array:
call dst(ny,nx,rvar,xtrig,xfactors)

return
end subroutine

!===================================================================

subroutine xderiv_ss(nx,ny,rkx,var,der)
! Computes der = d(var)/dx, spectrally, for a variable var(nx,ny)
! represented by a  sine  series in x and a  sine  series in y.
! The array rkx contains the x wavenumbers.
! *** both var and der are spectral ***
! ==> der(0:nx,ny) changes to a cosine series in x
! @@@ the mean value of der (0 wavenumber) is assumed to be 0

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rkx(nx),var(nx,ny),der(0:nx,ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do ky=1,ny-1
  der(0 ,ky)=0.d0
   !The above implies the mean value of der for every ky is zero.
  do kx=1,nx-1
    der(kx,ky)=rkx(kx)*var(kx,ky)
  enddo
  der(nx,ky)=0.d0
   !der = 0 when kx = nx since var = 0 when kx = nx.
enddo

return
end subroutine

!===================================================================

subroutine yderiv_ss(nx,ny,rky,var,der)
! Computes der = d(var)/dy, spectrally, for a variable var(nx,ny)
! represented by a  sine  series in x and a  sine  series in y.
! The array rky contains the y wavenumbers.
! *** both var and der are spectral ***
! ==> der(nx,0:ny) changes to a cosine series in y
! @@@ the mean value of der (0 wavenumber) is assumed to be 0

implicit none

 !Argument declarations:
integer:: nx,ny
double precision:: rky(ny),var(nx,ny),der(nx,0:ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do kx=1,nx-1
  der(kx,0)=0.d0
   !The above implies the mean value of der for every kx is zero.
  der(kx,ny)=0.d0
   !der = 0 when ky = ny since var = 0 when ky = ny.
enddo
do ky=1,ny-1
  do kx=1,nx-1
    der(kx,ky)=rky(ky)*var(kx,ky)
  enddo
enddo

return
end subroutine

!===================================================================

subroutine xderiv_cs(nx,ny,rkx,var,der)
! Computes der = d(var)/dx, spectrally, for a variable var(0:nx,ny)
! represented by a cosine series in x and a  sine  series in y.
! The array rkx contains the x wavenumbers.
! *** both var and der are spectral ***
! ==> der(nx,ny) changes to a sine series in x

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rkx(nx),var(0:nx,ny),der(nx,ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do ky=1,ny-1
  do kx=1,nx-1
    der(kx,ky)=-rkx(kx)*var(kx,ky)
  enddo
  der(nx,ky)=0.d0
enddo

return
end subroutine

!===================================================================

subroutine yderiv_cs(nx,ny,rky,var,der)
! Computes der = d(var)/dy, spectrally, for a variable var(0:nx,ny)
! represented by a cosine series in x and a  sine  series in y.
! The array rky contains the y wavenumbers.
! *** both var and der are spectral ***
! ==> der(0:nx,0:ny) changes to a cosine series in y
! @@@ the mean value of der (0 wavenumber) is assumed to be 0

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rky(ny),var(0:nx,ny),der(0:nx,0:ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do kx=0,nx
  der(kx,0)=0.d0
   !The above implies the mean value of der for every kx is zero.
  der(kx,ny)=0.d0
   !der = 0 when ky = ny since var = 0 when ky = ny.
enddo
do ky=1,ny-1
  do kx=0,nx
    der(kx,ky)=rky(ky)*var(kx,ky)
  enddo
enddo

return
end subroutine

!===================================================================

subroutine xderiv_sc(nx,ny,rkx,var,der)
! Computes der = d(var)/dx, spectrally, for a variable var(nx,0:ny)
! represented by a  sine  series in x and a cosine series in y.
! The array rkx contains the x wavenumbers.
! *** both var and der are spectral ***
! ==> der(0:nx,0:ny) changes to a cosine series in x
! @@@ the mean value of der (0 wavenumber) is assumed to be 0

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rkx(nx),var(nx,0:ny),der(0:nx,0:ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do ky=0,ny
  der(0 ,ky)=0.d0
   !The above implies the mean value of der for every ky is zero.
  do kx=1,nx-1
    der(kx,ky)=rkx(kx)*var(kx,ky)
  enddo
  der(nx,ky)=0.d0
   !der = 0 when kx = nx since var = 0 when kx = nx.
enddo

return
end subroutine

!===================================================================

subroutine yderiv_sc(nx,ny,rky,var,der)
! Computes der = d(var)/dy, spectrally, for a variable var(nx,0:ny)
! represented by a  sine  series in x and a cosine series in y.
! The array rky contains the y wavenumbers.
! *** both var and der are spectral ***
! ==> der(nx,ny) changes to a sine series in y

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rky(ny),var(nx,0:ny),der(nx,ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do kx=1,nx-1
  der(kx,ny)=0.d0
enddo
do ky=1,ny-1
  do kx=1,nx-1
    der(kx,ky)=-rky(ky)*var(kx,ky)
  enddo
enddo

return
end subroutine

!===================================================================

subroutine xderiv_cc(nx,ny,rkx,var,der)
! Computes der = d(var)/dx, spectrally, for a variable var(0:nx,0:ny)
! represented by a cosine series in x and a cosine series in y.
! The array rkx contains the x wavenumbers.
! *** both var and der are spectral ***
! ==> der(nx,0:ny) changes to a sine series in x

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rkx(nx),var(0:nx,0:ny),der(nx,0:ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do ky=0,ny
  do kx=1,nx-1
    der(kx,ky)=-rkx(kx)*var(kx,ky)
  enddo
  der(nx,ky)=0.d0
enddo

return
end subroutine

!===================================================================

subroutine yderiv_cc(nx,ny,rky,var,der)
! Computes der = d(var)/dy, spectrally, for a variable var(0:nx,0:ny)
! represented by a cosine series in x and a cosine series in y.
! The array rky contains the y wavenumbers.
! *** both var and der are spectral ***
! ==> der(0:nx,ny) changes to a sine series in y

implicit none

 !Arguments declarations:
integer:: nx,ny
double precision:: rky(ny),var(0:nx,0:ny),der(0:nx,ny)
 !Local declarations:
integer:: kx,ky

 !Carry out differentiation by wavenumber multiplication:
do kx=0,nx
  der(kx,ny)=0.d0
enddo
do ky=1,ny-1
  do kx=0,nx
    der(kx,ky)=-rky(ky)*var(kx,ky)
  enddo
enddo

return
end subroutine

!===================================================================


end module
