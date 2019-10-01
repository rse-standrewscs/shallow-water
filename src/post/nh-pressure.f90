!#########################################################################
!  Computes the rms non-hydrostatic pressure pn relative to the rms total
!  pressure p = g(h-z) + pn and writes the data to pn_rms.asc

!           Written 20/5/2019 by D G Dritschel @ St Andrews
!#########################################################################

program nhpressure

 !Import constants:
use constants

implicit none

 !Various arrays needed below:
double precision:: pn(ng,ng,0:nz)
double precision:: ph(ng,ng)
double precision:: theta(0:nz),weight(0:nz)
double precision:: prms,pnrms

 !Other local variables:
real:: tr4,q2dr4(ng,ng),q3dr4(ng,ng,0:nz)
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
open(31,file='3d/pn.r4',form='unformatted',access='direct', &
                        status='old',recl=ntbytes)

open(44,file= '2d/h.r4',form='unformatted',access='direct', &
                      status='old',recl=nhbytes)

 !Open output file:
open(61,file='pn_rms.asc',status='replace')

!---------------------------------------------------------------
 !Read data and process:
loop=0
do  
  loop=loop+1
  iread=0
  read(31,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  pn=dble(q3dr4)

  read(44,rec=loop,iostat=iread) tr4,q2dr4
  if (iread .ne. 0) exit 
  ph=csq*dble(q2dr4)

  write(*,'(a,f9.2)') ' *** Processing t = ',tr4

  !Compute relative rms error:
  prms=zero
  pnrms=zero
  do iz=0,nz
    prms=prms+weight(iz)*sum((ph+pn(:,:,iz))**2)
    pnrms=pnrms+weight(iz)*sum(pn(:,:,iz)**2)
  enddo
  pnrms=sqrt(pnrms/prms)

  !Write data for this time:
  write(61,'(1x,f12.5,1x,e14.7)') tr4,pnrms

enddo

 !Close files:
close(31)
close(44)
close(61)

write(*,*)
write(*,*) ' The relative rms non-hydrostatic pressure is in pn_rms.asc'

 !End main program
end program nhpressure
!=======================================================================
