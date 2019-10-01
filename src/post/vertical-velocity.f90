!#########################################################################
!  Extracts the vertical velocity w at z = h and writes w/H to 2d/w.r4

!  Also computes the rms difference between w(x,y,z,t) and w(x,y,h,t)*z/h
!  normalised by the rms w and writes the data to dw_rms.asc

!           Written 13/5/2019 by D G Dritschel @ St Andrews
!#########################################################################

program vvel

 !Import constants:
use constants

implicit none

 !Various arrays needed below:
double precision:: w(ng,ng,0:nz),z(ng,ng,0:nz),r(ng,ng,0:nz)
double precision:: w2d(ng,ng),wkp(ng,ng)
double precision:: theta(0:nz),weight(0:nz)
double precision:: wrms,dwrms

 !Other local variables:
real:: tr4,q3dr4(ng,ng,0:nz)
integer:: loop,iread,iz

!---------------------------------------------------------------
 !Define theta and the vertical weight for trapezoidal integration:
do iz=0,nz
  theta(iz)=dz*dble(iz)
  weight(iz)=one
enddo
weight(0)=f12
weight(nz)=f12
weight=weight/dble(nz)

!---------------------------------------------------------------
 !Open input data files:
open(31,file='3d/w.r4',form='unformatted',access='direct', &
                       status='old',recl=ntbytes)

open(32,file='3d/r.r4',form='unformatted',access='direct', &
                       status='old',recl=ntbytes)

 !Open output files:
open(51,file='2d/w.r4',form='unformatted',access='direct', &
                       status='replace',recl=nhbytes)

open(61,file='dw_rms.asc',status='replace')

!---------------------------------------------------------------
 !Read data and process:
loop=0
do  
  loop=loop+1
  iread=0
  read(31,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  w=dble(q3dr4)

  read(32,rec=loop) tr4,q3dr4
  r=dble(q3dr4)

  write(*,'(a,f9.2)') ' *** Processing t = ',tr4

  !Find z by trapezoidal integration of rho_theta (integrate over
  !rho'_theta then add theta to the result):
  z(:,:,1)=dz2*(r(:,:,0)+r(:,:,1))
  do iz=1,nz-1
    z(:,:,iz+1)=z(:,:,iz)+dz2*(r(:,:,iz)+r(:,:,iz+1))
  enddo
  do iz=1,nz
    z(:,:,iz)=z(:,:,iz)+theta(iz)
  enddo

  !Extract w/H at z = h:
  w2d=w(:,:,nz)*hinv

  !Write data:
  write(51,rec=loop) tr4,real(w2d)

  !Compute relative rms error:
  wrms=zero
  dwrms=zero
  wkp=w(:,:,nz)/z(:,:,nz)
  do iz=1,nz
    wrms=wrms+weight(iz)*sum(w(:,:,iz)**2)
    dwrms=dwrms+weight(iz)*sum((w(:,:,iz)-wkp*z(:,:,iz))**2)
  enddo
  dwrms=sqrt(dwrms/wrms)

  !Write data for this time:
  write(61,'(1x,f12.5,1x,e14.7)') tr4,dwrms

enddo

 !Close files:
close(31)
close(32)
close(51)
close(61)

write(*,*)
write(*,*) ' w/H at z = h is written to 2d/w.r4'
write(*,*) ' and the relative rms departure of w from a linear function'
write(*,*) ' is written to dw_rms.asc'


 !End main program
end program vvel
!=======================================================================
