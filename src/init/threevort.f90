program v3
!-----------------------------------------------------------------
!    Generates three Gaussian PV anomalies with zero divergence
!    and acceleration divergence.
!-----------------------------------------------------------------

use constants

implicit none

double precision:: qq(ng,ng)
double precision:: a1,x1,y1,r1,s1
double precision:: a2,x2,y2,r2,s2
double precision:: a3,x3,y3,r3,s3
double precision:: xg,yg,xoff,yoff
integer:: i,j,ix,iy

write(*,*) ' PV anomaly/f for vortex 1? '
read(*,*) a1
write(*,*) ' x_1 relative to domain width? '
read(*,*) x1
write(*,*) ' y_1 relative to domain width? '
read(*,*) y1
write(*,*) ' R_1 relative to domain width? '
read(*,*) r1

write(*,*)
write(*,*) ' PV anomaly/f for vortex 2? '
read(*,*) a2
write(*,*) ' x_2 relative to domain width? '
read(*,*) x2
write(*,*) ' y_2 relative to domain width? '
read(*,*) y2
write(*,*) ' R_2 relative to domain width? '
read(*,*) r2

write(*,*)
write(*,*) ' PV anomaly/f for vortex 3? '
read(*,*) a3
write(*,*) ' x_3 relative to domain width? '
read(*,*) x3
write(*,*) ' y_3 relative to domain width? '
read(*,*) y3
write(*,*) ' R_3 relative to domain width? '
read(*,*) r3

! Redefine anomalies to include f:
a1=a1*cof
a2=a2*cof
a3=a3*cof

! Redefine positions and radii including domain width:
x1=x1*twopi-pi
y1=y1*twopi-pi
s1=f12/(r1*twopi)**2
x2=x2*twopi-pi
y2=y2*twopi-pi
s2=f12/(r2*twopi)**2
x3=x3*twopi-pi
y3=y3*twopi-pi
s3=f12/(r3*twopi)**2

! Generate PV anomaly field and enforce periodicity (use qq):
qq=zero
do j=-1,1
  xoff=twopi*dble(j)-pi
  do i=-1,1
    yoff=twopi*dble(i)-pi
    do ix=1,ng
      xg=gl*dble(ix-1)+xoff
      do iy=1,ng
        yg=gl*dble(iy-1)+yoff
        qq(iy,ix)=qq(iy,ix)+a1*exp(-s1*((xg-x1)**2+(yg-y1)**2))+ &
                            a2*exp(-s2*((xg-x2)**2+(yg-y2)**2))+ &
                            a3*exp(-s3*((xg-x3)**2+(yg-y3)**2))
      enddo
    enddo
  enddo
enddo

! Write PV:
open(11,file='qq_init.r8',form='unformatted', &
    & access='direct',status='replace',recl=2*nhbytes)
write(11,rec=1) zero,qq
close(11)

! Write zero divergence and acceleration divergence (geostrophic balance):
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
