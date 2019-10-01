program energy
!  -----------------------------------------------------------------------
!  |   Computes the various components making up the total energy from   |
!  |   data in 3d/ql.r4, 3d/d.r4, 3d/g.r4, 3d/w.r4 and 2d/h.r4           |
!  -----------------------------------------------------------------------

 !Import spectral module:
use spectral

implicit none

!---------------------------------------------------------
 !Initialise inversion constants and arrays:
call init_spectral

 !Read data and process:
call diagnose

 !Internal subroutine definitions (inherit global variables):

contains 

!=======================================================================

subroutine diagnose

implicit none

 !Various arrays needed below:
double precision:: u(ng,ng,0:nz),v(ng,ng,0:nz),w(ng,ng,0:nz)
double precision:: r(ng,ng,0:nz),z(ng,ng,0:nz),zeta(ng,ng,0:nz)
double precision:: qs(ng,ng,0:nz),ds(ng,ng,0:nz),gs(ng,ng,0:nz)

double precision:: h2d(ng,ng),wkp(ng,ng)

 !Other local variables:
real:: tr4,q2dr4(ng,ng),q3dr4(ng,ng,0:nz)

 !Diagnostic quantities:
double precision:: ekin,epot,etot
integer:: loop,iread,iz

!---------------------------------------------------------------
 !Open input data files:
open(31,file='3d/ql.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(32,file= '3d/d.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(33,file= '3d/g.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(35,file= '3d/w.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(44,file= '2d/h.r4',form='unformatted',access='direct', &
                      status='old',recl=nhbytes)

 !Open output file:
open(22,file='e.asc',status='replace')

!---------------------------------------------------------------
 !Read data and process:
loop=0
do  
  loop=loop+1
  iread=0
  read(31,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  zeta=dble(q3dr4)
  call ptospc3d(zeta,qs,0,nz)

  read(32,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  zeta=dble(q3dr4)
  call ptospc3d(zeta,ds,0,nz)

  read(33,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  zeta=dble(q3dr4)
  call ptospc3d(zeta,gs,0,nz)

  read(35,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  w=dble(q3dr4)

  read(44,rec=loop,iostat=iread) tr4,q2dr4
  if (iread .ne. 0) exit 
  h2d=dble(q2dr4)

  !Obtain velocity field by inversion:
  call main_invert(qs,ds,gs,r,u,v,zeta)
   !Note: qs, ds & gs are in spectral space while 
   !      r, u, v and zeta are in physical space.

  !Find z by trapezoidal integration of rho_theta (integrate over
  !rho'_theta then add theta to the result):
  z(:,:,1)=dz2*(r(:,:,0)+r(:,:,1))
  do iz=1,nz-1
    z(:,:,iz+1)=z(:,:,iz)+dz2*(r(:,:,iz)+r(:,:,iz+1))
  enddo
  do iz=1,nz
    z(:,:,iz)=z(:,:,iz)+theta(iz)
  enddo

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
  write(22,'(f13.6,5(1x,f16.9))') tr4,zero,ekin,ekin,epot,etot
  write(*,'(a,f13.6,a,f13.6)') ' t = ',tr4,'  E_tot = ',etot
enddo

 !Close files:
close(22)
close(31)
close(32)
close(33)
close(35)
close(44)

write(*,*)
write(*,*) ' t vs E_div, E_kin, E_kin+E_div, E_pot & E_tot are ready in e.asc.'

return
end subroutine

 !End main program
end program
!=======================================================================
