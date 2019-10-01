program strip
! Initialises a PV strip with zero fields of divergence and 
! acceleration divergence.

use constants

implicit double precision(a-h,o-z)

integer,parameter:: ngu=16*ng
double precision,parameter:: qmax=4.d0*pi
double precision:: qod0(ngu/2),qod1(ngu/2),qod2(ngu/2)
double precision:: qev0(0:ngu/2),qev1(0:ngu/2),qev2(0:ngu/2)
double precision:: qa(ngu,ngu),qq(ng,ng)

write(*,*) ' Enter the width of the strip:'
read(*,*) wid
hwid=wid/two

write(*,*) ' The upper edge is displaced by A_2*sin(2kx)+A_3*sin(3kx) where'
write(*,*) ' k = 2*pi/L_x.'
write(*,*) ' Enter A_2 & A_3:'
read(*,*) amp2,amp3

glu=twopi/dble(ngu)
do ix=1,ngu
  x=glu*dble(ix-1)-pi
  y1=-hwid
  y2=hwid+amp2*sin(2.d0*x)+amp3*sin(3.d0*x)
  do iy=1,ngu
    y=glu*dble(iy-1)-pi
    if ((y2-y)*(y-y1) .gt. zero) then
      qa(iy,ix)=4.d0*qmax*(y2-y)*(y-y1)/(y2-y1)**2
    else
      qa(iy,ix)=zero
    endif
  enddo
enddo

!------------------------------------------------------------------------
 !Average the PV field in qa to the coarser grid (ng,ng):
ngh=ngu
do while (ngh .gt. ng)
  nguf=ngh
  ngh=ngh/2
   !Perform nine-point averaging:
  do iy=1,ngh
    miy=2*iy
    qod2(iy)=qa(miy-1,nguf)
    qev2(iy)=qa(miy,nguf)
  enddo
  qev2(0)=qa(nguf,nguf)
  do ix=1,ngh
    mix=2*ix
    mixm1=mix-1
    do iy=1,ngh
      miy=2*iy
      qod1(iy)=qa(miy-1,mixm1)
      qod0(iy)=qa(miy-1,mix)
      qev1(iy)=qa(miy,mixm1)
      qev0(iy)=qa(miy,mix)
    enddo
    qev1(0)=qev1(ngh)
    qev0(0)=qev0(ngh)
    do iy=1,ngh
      qa(iy,ix)=0.0625d0*(qev0(iy)+qev0(iy-1)+qev2(iy)+qev2(iy-1)) &
                +0.125d0*(qev1(iy)+qev1(iy-1)+qod0(iy)+qod2(iy)) &
                  +0.25d0*qod1(iy)
    enddo
    do iy=1,ngh
      qod2(iy)=qod0(iy)
      qev2(iy)=qev0(iy)
    enddo
    qev2(0)=qev0(0)
  enddo
enddo

 !Calculate and remove average qq:
qavg=zero
do ix=1,ng
  do iy=1,ng
    qavg=qavg+qa(iy,ix)
  enddo
enddo
qavg=qavg/dble(ng*ng)

do ix=1,ng
  do iy=1,ng
    qq(iy,ix)=qa(iy,ix)-qavg
  enddo
enddo

open(11,file='qq_init.r8',form='unformatted', &
    & access='direct',status='replace',recl=2*nhbytes)
write(11,rec=1) zero,qq
close(11)

!-------------------------------------------------------------
! Write zero fields of divergence and acceleration divergence:
qq=zero
open(11,file='dd_init.r8',form='unformatted', &
    & access='direct',status='replace',recl=2*nhbytes)
write(11,rec=1) zero,qq
close(11)

open(11,file='gg_init.r8',form='unformatted', &
    & access='direct',status='replace',recl=2*nhbytes)
write(11,rec=1) zero,qq
close(11)

end program
