!#########################################################################
! Computes gamma-tilde = gamma + 2J(u,v) - 2*delta^2 and writes 2d/gt.r4

!      Also computes the 1D spectrum and writes gt-spectra.asc

!          Revised 23/8/2019 by D G Dritschel @ St Andrews
!#########################################################################

program gammat

 !Import spectral module:
use spectral

implicit none

 !Various arrays needed below:
double precision:: u(ng,ng,0:nz),v(ng,ng,0:nz),r(ng,ng,0:nz)
double precision:: d(ng,ng,0:nz),g(ng,ng,0:nz),zeta(ng,ng,0:nz)
double precision:: qs(ng,ng,0:nz),ds(ng,ng,0:nz),gs(ng,ng,0:nz)

double precision:: h2d(ng,ng),gt2d(ng,ng)
double precision:: wkp(ng,ng),wka(ng,ng)

 !Other local variables:
double precision:: gspec(0:ng),tmpspec(0:ng),zspec,dspec,ogspec,dk
real:: tr4,q2dr4(ng,ng),q3dr4(ng,ng,0:nz)
integer:: loop,iread,iz,k

!---------------------------------------------------------
 !Initialise inversion constants and arrays:
call init_spectral

!---------------------------------------------------------------
 !Open input data files:
open(31,file='3d/ql.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(32,file= '3d/d.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(33,file= '3d/g.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(44,file= '2d/h.r4',form='unformatted',access='direct', &
                      status='old',recl=nhbytes)
open(50,file='spectra.asc',status='old')

 !Open output file:
open(51,file='2d/gt.r4',form='unformatted',access='direct', &
                      status='replace',recl=nhbytes)
open(60,file='gt-spectra.asc',status='replace')

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
  d=dble(q3dr4)
  call ptospc3d(d,ds,0,nz)

  read(33,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  g=dble(q3dr4)
  call ptospc3d(g,gs,0,nz)

  read(44,rec=loop,iostat=iread) tr4,q2dr4
  if (iread .ne. 0) exit 
  h2d=dble(q2dr4)

  write(*,'(a,f9.2)') ' *** Processing t = ',tr4

   !Obtain velocity field by inversion:
  call main_invert(qs,ds,gs,r,u,v,zeta)
   !Note: qs, ds & gs are in spectral space while 
   !      r, u, v and zeta are in physical space.

   !Compute vertically averaged gamma-tilde and its spectrum:
  gt2d=zero
  gspec=zero
  do iz=0,nz

     !Compute J(u,v):
    call jacob(u(:,:,iz),v(:,:,iz),wka)
    call spctop(ng,ng,wka,wkp,xfactors,yfactors,xtrig,ytrig)

     !Complete definition of gamma-tilde for this vertical grid point:
    wkp=g(:,:,iz)+two*(wkp-d(:,:,iz)**2)

     !De-alias and accumulate theta-averaged spectrum:
    call ptospc(ng,ng,wkp,wka,xfactors,yfactors,xtrig,ytrig)
    wka=filt*wka

    call spec1d(wka,tmpspec)
    gspec=gspec+weight(iz)*tmpspec

     !Return spectral gamma-tilde in wka to physical space as wkp:
    call spctop(ng,ng,wka,wkp,xfactors,yfactors,xtrig,ytrig)

     !Accumulate vertically-averaged gamma-tilde in physical space:
    gt2d=gt2d+weight(iz)*(one+r(:,:,iz))*wkp
  enddo

   !Normalise gt2d by dimensionless layer height:
  gt2d=gt2d/(one+h2d)
   !*** No further de-aliasing is done here or just above

   !Weight spectrum by spmf (see spectral.f90):
  gspec=spmf*gspec

   !Write data:
  write(51,rec=loop) tr4,real(gt2d)

  read(50,*)
  write(60,'(f9.2,1x,i5)') tr4,kmaxred
  do k=1,kmaxred
    read(50,*) dk,zspec,dspec,ogspec
    write(60,'(4(1x,f12.8))') alk(k),zspec,dspec,log10(gspec(k)+1.d-32)
  enddo

enddo

 !Close files:
close(31)
close(32)
close(33)
close(35)
close(44)
close(51)
close(50)
close(60)

write(*,*)
write(*,*) ' *** gamma-tilde written to 2d/gt.r4'
write(*,*) ' Spectra of zeta, delta & gamma-tilde written to gt-spectra.asc'

 !End main program
end program gammat
!=======================================================================
