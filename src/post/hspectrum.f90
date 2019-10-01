!#########################################################################
!  Computes the vertically-averaged thickness spectrum and writes 
!  alt_spectra.asc, for use by the plotting scripts.

!           Written 29/4/2019 by D G Dritschel @ St Andrews
!#########################################################################

program hspectrum

 !Import spectral module:
use spectral

implicit none

 !Physical array:
double precision:: r(ng,ng,0:nz)

 !Spectral array:
double precision:: rs(ng,ng,0:nz)

 !Other local variables:
double precision:: hspec(0:ng),tmpspec(0:ng),zspec,dspec,gspec,dk
real:: tr4,qqr4(ng,ng,0:nz)
integer:: loop,iread,k,iz

!----------------------------------------------------------------------
 !Initialise inversion constants and arrays:
call init_spectral

 !Open input data files:
open(34,file= '3d/r.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
open(50,file='spectra.asc',status='old')

 !Open output data files:
open(60,file='alt-spectra.asc',status='replace')

!----------------------------------------------------------------------
 !Read data and process:
loop=0
do
  loop=loop+1
  iread=0
  read(34,rec=loop,iostat=iread) tr4,qqr4
  if (iread .ne. 0) exit 
  r=dble(qqr4)
  call ptospc3d(r,rs,0,nz)

  write(*,'(a,f9.2)') ' *** Processing t = ',tr4

  !Compute 1d spectrum:
  hspec=zero
  do iz=0,nz
    call spec1d(rs(:,:,iz),tmpspec)
    hspec=hspec+weight(iz)*tmpspec
  enddo
  hspec=spmf*hspec

  read(50,*)
  write(60,'(f9.2,1x,i5)') tr4,kmaxred
  do k=1,kmaxred
    read(50,*) dk,zspec,dspec,gspec
    write(60,'(4(1x,f12.8))') alk(k),log10(hspec(k)+1.d-32),dspec,gspec
  enddo

enddo

 !Close files:
close(34)
close(50)
close(60)

write(*,*)
write(*,*) ' Spectra of r, delta & gamma are now in alt-spectra.asc.'

 !End main program
end program hspectrum
!=======================================================================
