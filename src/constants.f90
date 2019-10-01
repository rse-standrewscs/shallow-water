module constants

 !Module containing all non-modifiable parameters.

use parameters

 !Sizes of records used in unformatted writes of real*4 data:
integer,parameter:: nhgp=ng*ng,nzm1=nz-1
integer,parameter:: nhbytes=4*(nhgp+1),ntbytes=4*(nhgp*(nz+1)+1)

 !Generic double precision numerical constants:
double precision,parameter:: zero=0.d0,one=1.d0,two=2.d0,three=3.d0
double precision,parameter:: four=4.d0,five=5.d0,six=6.d0
double precision,parameter:: f12=one/two,f13=one/three,f23=two/three
double precision,parameter:: f14=one/four,f15=one/five,f16=one/six
double precision,parameter:: f56=five/six,f112=one/12.d0
double precision,parameter:: twopi=two*pi

 !Grid constants:
double precision,parameter:: domarea=twopi*twopi,dsumi=one/dble(ng*ng)
double precision,parameter:: gl=twopi/dble(ng),garea=gl*gl
double precision,parameter:: dz=hbar/dble(nz),dz2=dz/2.d0,dz6=dz/6.d0
double precision,parameter:: dzi=one/dz,dzisq=dzi**2
double precision,parameter:: hdzi=f12*dzi,qdzi=f14*dzi
double precision,parameter:: gvol=garea*dz,vsumi=one/dble(ng*ng*nz)

! Time step related parameters:
double precision,parameter:: dt2=dt*f12,dt4=dt*f14
double precision,parameter:: dt2i=one/dt2,dt4i=one/dt4

! Various physical constants:
double precision,parameter:: csq=cgw**2,fsq=cof**2,cofi=one/cof

 !Gravitational acceleration and inverse depth:
double precision,parameter:: gravity=csq/hbar,hinv=one/hbar

end module
