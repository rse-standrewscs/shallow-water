program swto3d
! Converts 2D shallow-water fields to 3D fields needed by nhswps.

! Import parameters and constants:
use constants

implicit none

! Various arrays needed below:
double precision:: ql(ng,ng,0:nz),d(ng,ng,0:nz),g(ng,ng,0:nz)
double precision:: ql2d(ng,ng),d2d(ng,ng),g2d(ng,ng)
double precision:: t
integer:: iz

!---------------------------------------------------------
 !Read 2D shallow-water fields (e.g. generated by balinit.f90):
open(11,file='sw_init.r8',form='unformatted', &
      access='direct',status='old',recl=2*nhbytes)
read(11,rec=1) t,ql2d
read(11,rec=2) t,d2d
read(11,rec=3) t,g2d
close(11)

!---------------------------------------------------------
 !Take all variables to be independent of height:
do iz=0,nz
  ql(:,:,iz)=ql2d
  d(:,:,iz)=d2d
  g(:,:,iz)=g2d
enddo

!---------------------------------------------------------
 !Write 3D fields:
open(11,file='qq_init.r8',form='unformatted', &
      access='direct',status='replace',recl=2*ntbytes)
write(11,rec=1) t,ql
close(11)

open(11,file='dd_init.r8',form='unformatted', &
      access='direct',status='replace',recl=2*ntbytes)
write(11,rec=1) t,d
close(11)

open(11,file='gg_init.r8',form='unformatted', &
      access='direct',status='replace',recl=2*ntbytes)
write(11,rec=1) t,g
close(11)

write(*,*)
write(*,*) ' Initial 3d fields written to ql, dd and gg_init.r8'

end program swto3d