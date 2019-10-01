!#########################################################################
!  Computes the magnitude of the vertically-averaged non-hydrostatic 
!  acceleration and writes 2d/a.r4.

!  Computes also the rms non-hydrostatic acceleration relative to the 
!  rms total acceleration and writes an_rms.asc.

!           Written 30/4/2019 by D G Dritschel @ St Andrews
!#########################################################################

program accelnh

 !Import spectral module:
use spectral

implicit none

 !Various arrays needed below:
double precision:: r(ng,ng,0:nz),pn(ng,ng,0:nz)

double precision:: h2d(ng,ng)
double precision:: anx(ng,ng),any(ng,ng)
double precision:: ahx(ng,ng),ahy(ng,ng)
double precision:: wka(ng,ng),wkb(ng,ng),wke(ng,ng)

 !Other local variables:
double precision:: anrms
real:: tr4,q2dr4(ng,ng),q3dr4(ng,ng,0:nz)
integer:: loop,iread,iz

!---------------------------------------------------------
 !Initialise inversion constants and arrays:
call init_spectral

!---------------------------------------------------------------
 !Open input data files:
open(31,file='3d/r.r4' ,form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(36,file='3d/pn.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(44,file= '2d/h.r4',form='unformatted',access='direct', &
                      status='old',recl=nhbytes)

 !Open output files:
open(51,file= '2d/a.r4',form='unformatted',access='direct', &
                      status='replace',recl=nhbytes)
open(61,file='an_rms.asc',status='replace')

!---------------------------------------------------------------
 !Read data and process:
loop=0
do  
  loop=loop+1
  iread=0
  read(31,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  r=dble(q3dr4)

  read(36,rec=loop,iostat=iread) tr4,q3dr4
  if (iread .ne. 0) exit 
  pn=dble(q3dr4)

  read(44,rec=loop,iostat=iread) tr4,q2dr4
  if (iread .ne. 0) exit 
  h2d=dble(q2dr4)

  write(*,'(a,f9.2)') ' *** Processing t = ',tr4

  !Find vertically-averaged hydrostatic and non-hydrostatic acceleration:
  anx=zero
  any=zero
  ahx=zero
  ahy=zero
  do iz=0,nz-1
    !(P' = 0 when iz = nz)
    wkb=r(:,:,iz)
    call ptospc(ng,ng,wkb,wke,xfactors,yfactors,xtrig,ytrig)

    call xderiv(ng,ng,hrkx,wke,wka)
    call spctop(ng,ng,wka,wkb,xfactors,yfactors,xtrig,ytrig)
    anx=anx-weight(iz)*pn(:,:,iz)*wkb

    call yderiv(ng,ng,hrky,wke,wka)
    call spctop(ng,ng,wka,wkb,xfactors,yfactors,xtrig,ytrig)
    any=any-weight(iz)*pn(:,:,iz)*wkb
  enddo

  !Normalise to complete definition of vertically-averaged quantities:
  wkb=one/(one+h2d)
  anx=anx*wkb
  any=any*wkb

  !Next obtain hydrostatic acceleration (use dd & zz):
  call ptospc(ng,ng,h2d,wke,xfactors,yfactors,xtrig,ytrig)
  call xderiv(ng,ng,hrkx,wke,wka)
  call yderiv(ng,ng,hrky,wke,wkb)
  call spctop(ng,ng,wka,ahx,xfactors,yfactors,xtrig,ytrig)
  call spctop(ng,ng,wkb,ahy,xfactors,yfactors,xtrig,ytrig)
  ahx=-csq*ahx
  ahy=-csq*ahy

  !Form relative rms acceleration and write data:
  anrms=sqrt(sum(anx**2+any**2)/sum((ahx+anx)**2+(ahy+any)**2))
  write(61,'(1x,f12.5,1x,e14.7)') tr4,anrms

  !Form magnitude of acceleration and write data:
  anx=sqrt(anx**2+any**2)
  write(51,rec=loop) tr4,real(anx)

enddo

 !Close files:
close(31)
close(36)
close(44)
close(51)
close(61)

write(*,*)
write(*,*) ' Non-hydrostatic acceleration magnitude written to 2d/a.r4'
write(*,*) ' and its relative rms norm is written to ah_rms.asc'

 !End main program
end program accelnh
!=======================================================================
