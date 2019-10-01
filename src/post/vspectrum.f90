!#########################################################################
!  Computes the horizontally-averaged vertical spectrum of q_l, delta
!  and gamma at a specified time.  Write qdg_vspecnnnn.asc where nnnn
!  is 0000, 0001 ... - the corresponding time frame.

!           Written 8/7/2019 by D G Dritschel @ St Andrews
!#########################################################################

program vspectrum

 !Import 1D FFT module:
use stafft

 !Import constants:
use constants

implicit none
integer,parameter:: ngsq=ng*ng

 !Original fields to process:
double precision:: ql(ngsq,0:nz),d(ngsq,0:nz),g(ngsq,0:nz)

 !Other local variables:
double precision:: trig(2*nz)
integer:: factors(5)

double precision:: qspec(0:nz),dspec(0:nz),gspec(0:nz),fac
real:: tr4,q3dr4(ngsq,0:nz)
integer:: loop,i,m
character(len=17):: outfile

outfile='qdg_vspec0000.asc'

write(*,*) ' Enter the time you wish to analyse:'
read(*,*) tr4
loop=nint(tr4/tgsave)
write(outfile(10:13),'(i4.4)') loop

!----------------------------------------------------------------------
! Set up FFTs:
call initfft(nz,factors,trig)

 !Open input data files and read data:
open(31,file='3d/ql.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
read(31,rec=loop) tr4,q3dr4
ql=dble(q3dr4)
close(31)

open(31,file= '3d/d.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
read(31,rec=loop) tr4,q3dr4
d=dble(q3dr4)
close(31)

open(31,file= '3d/g.r4',form='unformatted',access='direct', &
                      status='old',recl=ntbytes)
read(31,rec=loop) tr4,q3dr4
g=dble(q3dr4)
close(31)

 !Perform cosine Fourier transforms (overwrites data by the coeffs):
call dct(ngsq,nz,ql,trig,factors)
call dct(ngsq,nz, d,trig,factors)
call dct(ngsq,nz, g,trig,factors)

qspec=zero
dspec=zero
gspec=zero
do m=0,nz
  do i=0,ngsq
    qspec(m)=qspec(m)+ql(i,m)**2
    dspec(m)=dspec(m)+ d(i,m)**2
    gspec(m)=gspec(m)+ g(i,m)**2
  enddo
enddo
fac=one/dble(ngsq)
qspec=fac*qspec
dspec=fac*dspec
gspec=fac*gspec
qspec(0)=f12*qspec(0)
dspec(0)=f12*dspec(0)
gspec(0)=f12*gspec(0)

open(60,file=outfile,status='replace')
do m=0,nz
  write(60,'(4(1x,f12.8))') dble(m),log10(qspec(m)+1.d-32), &
             log10(dspec(m)+1.d-32),log10(gspec(m)+1.d-32)
enddo
close(60)

write(*,*)
write(*,*) ' Spectra of q_l, delta & gamma are now in '//outfile

 !End main program
end program vspectrum
!=======================================================================
