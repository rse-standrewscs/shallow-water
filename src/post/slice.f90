!===============================================================
! Makes a image file looping over x = constant, y = constant or 
! z = constant slice of the data generated by nhswps.f90
!
! Note, data in z are averaged to the centres (in z) of each 
! grid box - i.e. they are shown on the half grid in z
!===============================================================

program slice

 ! Import parameters and constants:
use constants

implicit none

 !Local variables:
real:: f(ng,ng,0:nz),fx(nz,ng),fy(nz,ng),fz(ng,ng),t
integer:: iopt,iord,ix,iy,iz,irec
character(len=4):: pind
character(len=2):: pref

!----------------------------------------------------------
 !Ask for user inputs:
write(*,*) ' Which frame do you wish to view?'
read(*,*) irec
write(pind,'(i4.4)') irec

write(*,*) ' Choose on of the following fields:'
write(*,*) ' (1) d'
write(*,*) ' (2) g'
write(*,*) ' (3) pn'
write(*,*) ' (4) ql'
write(*,*) ' (5) r  or'
write(*,*) ' (6) w'
read(*,*) iopt

if (iopt .eq. 1) then
  pref='dd'
  open(81,file='3d/d.r4',form='unformatted',access='direct', &
                         status='old',recl=ntbytes)
else if (iopt .eq. 2) then
  pref='gg'
  open(81,file='3d/g.r4',form='unformatted',access='direct', &
                         status='old',recl=ntbytes)
else if (iopt .eq. 3) then
  pref='pn'
  open(81,file='3d/pn.r4',form='unformatted',access='direct', &
                         status='old',recl=ntbytes)
else if (iopt .eq. 4) then
  pref='ql'
  open(81,file='3d/ql.r4',form='unformatted',access='direct', &
                         status='old',recl=ntbytes)
else if (iopt .eq. 5) then
  pref='rr'
  open(81,file='3d/r.r4',form='unformatted',access='direct', &
                         status='old',recl=ntbytes)
else if (iopt .eq. 6) then
  pref='ww'
  open(81,file='3d/w.r4',form='unformatted',access='direct', &
                         status='old',recl=ntbytes)
else
  write(*,*) ' This option does not exist!'
  stop
endif

 !Read data:
read(81,rec=irec) t,f
close(81)

write(*,*)
write(*,'(a,f12.5)') ' Read data at t = ',t
write(*,*)

write(*,*) ' Image to loop over (1) ix, (2) iy or (3) iz?' 
read(*,*) iord

if (iord .eq. 1) then

  open(31,file='3d/'//pref//pind//'_x.r4',form='unformatted', &
          status='replace',access='direct',recl=4*(ng*nz+1))
  do ix=1,ng
    do iy=1,ng
      do iz=1,nz
        fx(iz,iy)=0.5*(f(iy,ix,iz-1)+f(iy,ix,iz))
      enddo
    enddo
    write(31,rec=ix) real(ix),fx
  enddo
  close(31)
  write(*,*)
  write(*,*) ' To image the file, type'
  write(*,*)
  write(*,'(a,2(i3,1x),a)') &
   ' dataview -ndim ',ng,nz,'3d/'//pref//pind//'_x.r4 -glob &'
  write(*,*)

else if (iord .eq. 2) then

  open(31,file='3d/'//pref//pind//'_y.r4',form='unformatted', &
          status='replace',access='direct',recl=4*(ng*nz+1))
  do iy=1,ng
    do ix=1,ng
      do iz=1,nz
        fy(iz,ix)=0.5*(f(iy,ix,iz-1)+f(iy,ix,iz))
      enddo
    enddo
    write(31,rec=iy) real(iy),fy
  enddo
  close(31)
  write(*,*)
  write(*,*) ' To image the file, type'
  write(*,*)
  write(*,'(a,2(i3,1x),a)') &
   ' dataview -ndim ',ng,nz,'3d/'//pref//pind//'_y.r4 -glob &'
  write(*,*)

else

  open(31,file='3d/'//pref//pind//'_z.r4',form='unformatted', &
          status='replace',access='direct',recl=4*(ng*ng+1))
  do iz=1,nz
    do ix=1,ng
      do iy=1,ng
        fz(iy,ix)=0.5*(f(iy,ix,iz-1)+f(iy,ix,iz))
      enddo
    enddo
    write(31,rec=iz) real(iz),fz
  enddo
  close(31)
  write(*,*)
  write(*,*) ' To image the file, type'
  write(*,*)
  write(*,'(a,2(i3,1x),a)') &
   ' dataview -ndim ',ng,ng,'3d/'//pref//pind//'_z.r4 -glob &'
  write(*,*)

endif

 !End main program
end program
