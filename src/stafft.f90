module stafft

implicit none

! Fourier transform module. 
! This is not a general purpose transform package but is designed to be 
! quick for arrays of length 2^n. It will work if the array length is of
! the form 2^i * 3^j * 4^k * 5^l * 6^m (integer powers obviously).
! 
! Minimal error-checking is performed by the code below. The only check is that 
! the initial factorisation can be performed. 
! Therefore if the transforms are called with an array of length <2, or a trig array 
! not matching the length of the array to be transformed the code will fail in a 
! spectacular way (eg. Seg. fault or nonsense returned). 
! It is up to the calling code to ensure everything is called sensibly. 
! The reason for stripping error checking is to speed up the backend by performing
! less if() evaluations - as errors in practice seem to occur very rarely. 
! So the good news is this should be a fast library - the bad is that you may have to pick
! around in it if there are failures.
!
! To initialise the routines call init(n,factors,trig,ierr).
! This fills a factorisation array (factors), and a sin/cos array (trig).
! These must be kept in memory by the calling program.
! The init routine can be called multiple times with different arrays if more than 
! one length of array is to be transformed. 
! If a factorisation of the array length n cannot be found (as specified above) 
! then the init routine will exit immediately and the integer ierr will be set to 1.
! If the init returns with ierr=0 then the call was successful.
!
! Top-level subroutines contained in this module are:
! 1) initfft(n,factors,trig)        :
!      Performs intialisation of the module, by working out the factors of n (the FFT length).
!      This will fail if n is not factorised completely by 2,3,4,5,6.
!      The trig array contains the necessary cosine and sine values. 
!      Both arrays passed to init **must** be kept between calls to routines in this module. 
! 2) forfft(m,n,x,trig,factors)  :
!      This performs a FFT of an array x containing m vectors of length n.
!      The transform length is n.
!      This inverse of this transform is obtained by revfft.
! 3) revfft(m,n,x,trig,factors)  :
!      This performs an inverse FFT of an array x containing m vectors of length n.
!      The transform length is n.
!      This inverse of this transform is forfft.
! 4) dct(m,n,x,trig,factors)     :
!      This performs a discrete cosine transform of an array x containing m vectors of length n.
!      The transform length is n.
!      This routine calls forfft and performs pre- and post- processing to obtain the transform.
!      This transform is it's own inverse.
! 5) dst(m,n,x,trig,factors)     :
!      This performs a discrete sine transform of an array x containing m vectors of length n.
!      The transform length is n.
!      This routine calls forfft and performs pre- and post- processing to obtain the transform.
!      This transform is it's own inverse.
!
! The storage of the transformed array is in 'Hermitian form'. This means that, for the jth vector 
! the values x(j,1:nw) contain the cosine modes of the transform, while the values x(j,nw+1:n) contain
! the sine modes (in reverse order ie. wave number increasing from n back to nw+1).
! [Here, for even n, nw=n/2, and for odd n, nw=(n-1)/2].


contains
!----------------------------
subroutine initfft(n,factors,trig)

! Subroutine performs initialisation work for all the transforms.
! It calls routines to factorise the array length n and then sets up
! a trig array full of sin/cos values used in the transform backend. 

implicit none

 !Argument declarations:
integer:: n,factors(5)
double precision:: trig(2*n)

 !Local declarations:
double precision,parameter:: twopi=6.283185307179586476925286766559005768394338798750211641949889184615633d0
double precision:: ftwopin
integer:: i,j,k,l,m,fac(5),rem,ierr
!---------------------------------

 !First factorise n:
call factorisen(n,factors,ierr)

 !Return if factorisation unsuccessful:
if (ierr .eq. 1) then 
   !Catastrophic end to run if factorisation fails:
  write(*,*) '****************************'
  write(*,*) ' Factorisation not possible.'
  write(*,*) ' Only factors from 2-6 allowed.'
  write(*,*) ' STOPPING...'
  write(*,*) '****************************'
  stop
endif

 !Define list of factors array:
fac(1)=6
fac(2)=4
fac(3)=2
fac(4)=3
fac(5)=5

 !Define constants needed in trig array definition: 
ftwopin=twopi/dble(n)
rem=n
m=1
do i=1,5
  do j=1,factors(i)
    rem=rem/fac(i)
    do k=1,fac(i)-1
      do l=0,rem-1
        trig(m)=ftwopin*dble(k*l)
        m=m+1
      enddo
    enddo
    ftwopin=ftwopin*fac(i)
  enddo
enddo

do i=1,n-1 
  trig(n+i)=-sin(trig(i))
  trig(i)  = cos(trig(i))
enddo

return 
end subroutine
!============================================

subroutine factorisen(n,factors,ierr)
      
implicit none

 !Argument declarations: 
integer:: n,factors(5),ierr
 !Local declarations:
integer:: i,rem
!--------------------------------

ierr=0
 !Initialiase factors array:
do i=1,5
  factors(i)=0
enddo

rem=n
 !Find factors of 6: 
do while (mod(rem,6) .eq. 0) 
  factors(1)=factors(1)+1
  rem=rem/6
  if (rem .eq. 1) return
enddo

 !Find factors of 4:
do while (mod(rem,4) .eq. 0) 
  factors(2)=factors(2)+1
  rem=rem/4
  if (rem .eq. 1) return
enddo

 !Find factors of 2:
do while (mod(rem,2) .eq. 0) 
  factors(3)=factors(3)+1
  rem=rem/2
  if (rem .eq. 1) return
enddo

 !Find factors of 3:
do while (mod(rem,3) .eq. 0) 
  factors(4)=factors(4)+1
  rem=rem/3
  if (rem .eq. 1) return
enddo

 !Find factors of 5:
do while (mod(rem,5) .eq. 0) 
  factors(5)=factors(5)+1
  rem=rem/5
  if (rem .eq. 1) return
enddo
 !If code reaches this point factorisation has 
 !failed - return error code in ierr:  
ierr=1

return
end subroutine
!============================================

subroutine forfft(m,n,x,trig,factors)

! Main physical to spectral (forward) FFT routine. 
! Performs m transforms of length n in the array x which is dimensioned x(m,n).
! The arrays trig and factors are filled by the init routine and 
! should be kept from call to call.
! Backend consists of mixed-radix routines, with 'decimation in time'.
! Transform is stored in Hermitian form.

implicit none

 !Arguments declarations:
double precision:: x(0:m*n-1),trig(0:2*n-1) 
integer:: m,n,factors(5)

 !Local declarations:
double precision:: wk(0:m*n-1),normfac
integer:: i,rem,cum,iloc
logical:: orig

!--------------------------------------
 !Initialise flip/flop logical and counters
orig=.true.
rem=n
cum=1
 !Use factors of 5:
do i=1,factors(5)
  rem=rem/5
  iloc=(rem-1)*5*cum
  if (orig) then
    call forrdx5(x,wk,m*rem,cum,trig(iloc),trig(n+iloc))
  else 
    call forrdx5(wk,x,m*rem,cum,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*5
enddo
 !Use factors of 3:
do i=1,factors(4)
  rem=rem/3
  iloc=(rem-1)*3*cum
  if (orig) then
    call forrdx3(x,wk,m*rem,cum,trig(iloc),trig(n+iloc))
  else 
    call forrdx3(wk,x,m*rem,cum,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*3
enddo
 !Use factors of 2:
do i=1,factors(3)
  rem=rem/2
  iloc=(rem-1)*2*cum
  if (orig) then
    call forrdx2(x,wk,m*rem,cum,trig(iloc),trig(n+iloc))
  else 
    call forrdx2(wk,x,m*rem,cum,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*2
enddo
 !Use factors of 4:
do i=1,factors(2)
  rem=rem/4
  iloc=(rem-1)*4*cum
  if (orig) then
    call forrdx4(x,wk,m*rem,cum,trig(iloc),trig(n+iloc))
  else 
    call forrdx4(wk,x,m*rem,cum,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*4
enddo
 !Use factors of 6:
do i=1,factors(1)
  rem=rem/6
  iloc=(rem-1)*6*cum
  if (orig) then
    call forrdx6(x,wk,m*rem,cum,trig(iloc),trig(n+iloc))
  else 
    call forrdx6(wk,x,m*rem,cum,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*6
enddo

 !Multiply by the normalisation constant and put
 !transformed array in the right location:
normfac=1.0d0/sqrt(dble(n))
if (orig) then
  do i=0,m*n-1
    x(i)=x(i)*normfac
  enddo
else
  do i=0,m*n-1
    x(i)=wk(i)*normfac
  enddo
endif

return
end subroutine
!=====================================================

subroutine revfft(m,n,x,trig,factors)

! Main spectral to physical (reverse) FFT routine. 
! Performs m reverse transforms of length n in the array x which is dimensioned x(m,n).
! The arrays trig and factors are filled by the init routine and 
! should be kept from call to call.
! Backend consists of mixed-radix routines, with 'decimation in frequency'.
! Reverse transform starts in Hermitian form.

implicit none

 !Arguments declarations:
double precision:: x(0:m*n-1),trig(0:2*n-1) 
integer:: m,n,factors(5)

 !Local declarations:
double precision:: wk(0:m*n-1),normfac
integer:: i,k,cum,rem,iloc
logical:: orig

!----------------------------------------
 !Flip the sign of the sine coefficients:
do i=(n/2+1)*m,n*m-1
  x(i)=-x(i)
enddo

 !Scale 0 and Nyquist frequencies:
do i=0,m-1
  x(i)=0.5d0*x(i)
enddo  
if (mod(n,2) .eq. 0) then
  k=m*n/2
  do i=0,m-1
    x(k+i)=0.5d0*x(k+i)
  enddo
endif

 !Initialise flip/flop logical and counters
orig=.true.
cum=1
rem=n
 !Use factors of 6:
do i=1,factors(1)
  rem=rem/6
  iloc=(cum-1)*6*rem
  if (orig) then
    call revrdx6(x,wk,m*cum,rem,trig(iloc),trig(n+iloc))
  else 
    call revrdx6(wk,x,m*cum,rem,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*6
enddo
 !Use factors of 4:
do i=1,factors(2)
  rem=rem/4
  iloc=(cum-1)*4*rem
  if (orig) then
    call revrdx4(x,wk,m*cum,rem,trig(iloc),trig(n+iloc))
  else 
    call revrdx4(wk,x,m*cum,rem,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*4
enddo
 !Use factors of 2:
do i=1,factors(3)
  rem=rem/2
  iloc=(cum-1)*2*rem
  if (orig) then
    call revrdx2(x,wk,m*cum,rem,trig(iloc),trig(n+iloc))
  else 
    call revrdx2(wk,x,m*cum,rem,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*2
enddo
 !Use factors of 3:
do i=1,factors(4)
  rem=rem/3
  iloc=(cum-1)*3*rem
  if (orig) then
    call revrdx3(x,wk,m*cum,rem,trig(iloc),trig(n+iloc))
  else 
    call revrdx3(wk,x,m*cum,rem,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*3
enddo
 !Use factors of 5:
do i=1,factors(5)
  rem=rem/5
  iloc=(cum-1)*5*rem
  if (orig) then
    call revrdx5(x,wk,m*cum,rem,trig(iloc),trig(n+iloc))
  else 
    call revrdx5(wk,x,m*cum,rem,trig(iloc),trig(n+iloc))
  endif
  orig=.not. orig
  cum=cum*5
enddo

 !Multiply by the normalisation constant and put
 !transformed array in the right location:
normfac=2.0d0/sqrt(dble(n))
if (orig) then
  do i=0,m*n-1
    x(i)=x(i)*normfac
  enddo
else
  do i=0,m*n-1
    x(i)=wk(i)*normfac
  enddo
endif

return
end subroutine
!============================================

subroutine dct(m,n,x,trig,factors)

! This routine computes multiple fourier cosine transforms of sequences
! of doubles using the forfft routine to compute the FFT,
! along with pre- and post-processing steps to extract the dst.

implicit none

!Argument declarations:
integer:: m,n,factors(5)
double precision:: x(m,0:n),trig(2*n)

!Local declarations:
double precision,parameter:: pi=3.141592653589793238462643383279502884197169399375105820974944592307816d0
double precision,parameter:: rt2=1.414213562373095048801688724209698078569671875376948073176679737990732d0
double precision:: wk(1:m,0:n-1),fpin,rtn,rowsum
integer:: i,j,nd2

!--------------------------------------------------
fpin=pi/dble(n)
rtn=sqrt(dble(n))

 !Pre-process the array and store it in wk:
do i=1,m
  wk(i,0)=0.5d0*(x(i,0)+x(i,n))
enddo

do j=1,n-1
  do i=1,m
    wk(i,j)=0.5d0*(x(i,j)+x(i,n-j))-sin(dble(j)*fpin)*(x(i,j)-x(i,n-j))
  enddo
enddo

 !Get the first element of the transform x(i,1) and store 
 !in x(i,n), as this is not overwritten when x is used
 !as a work array in the forfft routine called next: 
do i=1,m
  rowsum=0.0d0
  rowsum=rowsum+0.5d0*x(i,0)
  do j=1,n-1
    rowsum=rowsum+x(i,j)*cos(dble(j)*fpin)
  enddo
  rowsum=rowsum-0.5d0*x(i,n)
  x(i,n)=rt2*rowsum/rtn
enddo

 !Transform the wk array by use of the general FFT routine:
call forfft(m,n,wk,trig,factors)

 !Post-process the result of the FFT to get the dst of x and
 !put the result back into the x array:
do i=1,m
  x(i,0)=rt2*wk(i,0)
enddo
do i=1,m
  x(i,1)=x(i,n)
enddo

if (mod(n,2) .eq. 0) then
  nd2=n/2
  do j=1,nd2-1
    do i=1,m
      x(i,2*j)=rt2*wk(i,j)
      x(i,2*j+1)=x(i,2*j-1)-rt2*wk(i,n-j)
    enddo
  enddo
  do i=1,m
    x(i,n)=rt2*wk(i,nd2)
  enddo
else if (mod(n,2) .eq. 1) then
  do j=1,(n-1)/2
    do i=1,m
      x(i,2*j)=rt2*wk(i,j)
      x(i,2*j+1)=x(i,2*j-1)-rt2*wk(i,n-j)
    enddo
  enddo
endif

return
end subroutine
!=============================================================

subroutine dst(m,n,x,trig,factors)

! This routine computes multiple fourier sine transforms of sequences
! of doubles using the forfft routine to compute the FFT,
! along with pre- and post-processing steps to extract the dst.

implicit none

!Argument declarations:
integer:: m,n,factors(5)
double precision:: x(m,n),trig(2*n)

!Local declarations:
double precision,parameter:: pi=3.141592653589793238462643383279502884197169399375105820974944592307816d0
double precision,parameter:: rt2=1.414213562373095048801688724209698078569671875376948073176679737990732d0
double precision:: wk(1:m,0:n-1),fpin
integer:: i,j

!------------------------------------------
fpin=pi/dble(n)

 !Pre-process the array and store it in wk:
 !First set 0 frequency element to zero:
do i=1,m
  wk(i,0)=0.0d0
enddo

 !Next set up the rest of the array:
do j=1,n-1
  do i=1,m
    wk(i,j)=0.5d0*(x(i,j)-x(i,n-j))+sin(dble(j)*fpin)*(x(i,j)+x(i,n-j))
  enddo
enddo

 !Transform the wk array by use of the general FFT routine:
call forfft(m,n,wk,trig,factors)

 !Post-process the result of the FFT to get the dst of x and
 !put the result back into the x array:
do i=1,m
  x(i,1)=wk(i,0)/rt2
enddo
if (mod(n,2) .eq. 0) then
  do j=1,n/2-1
    do i=1,m
      x(i,2*j)=-rt2*wk(i,n-j)
    enddo
    do i=1,m
      x(i,2*j+1)=rt2*wk(i,j)+x(i,2*j-1)
    enddo
  enddo
else if (mod(n,2) .eq. 1) then
  do j=1,(n-1)/2-1
    do i=1,m
      x(i,2*j)=-rt2*wk(i,n-j)
      x(i,2*j+1)=rt2*wk(i,j)+x(i,2*j-1)
    enddo
  enddo
  do i=1,m
    x(i,n-1)=-rt2*wk(i,(n+1)/2)
  enddo
endif

 !  Set the Nyquist frequency element to zero:
do i=1,m
  x(i,n)=0.0d0
enddo

return
end subroutine 
!==================================================

!====================================================
!  Internal radix routines only beyond this point...
!  Abandon hope all ye who enter in!
!====================================================
! Physical to spectral (forward) routines:
!====================================================	
subroutine forrdx6(a,b,nv,lv,cosine,sine)

! Radix six physical to Hermitian FFT with 'decimation in time'.

implicit none
      
 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:5,0:lv-1),b(0:nv-1,0:lv-1,0:5),cosine(0:lv-1,5),sine(0:lv-1,5)

 !Local declarations:
double precision,parameter:: sinfpi3=0.8660254037844386467637231707529361834714026269051903140279034897259665d0
double precision:: x1p,x2p,x3p,x4p,x5p
double precision:: y1p,y2p,y3p,y4p,y5p
double precision:: s1k,s2k,s3k,s4k,s5k
double precision:: c1k,c2k,c3k,c4k,c5k
double precision:: t1i,t1r,t2i,t2r,t3i,t3r
double precision:: u0i,u0r,u1i,u1r,u2i,u2r
double precision:: v0i,v0r,v1i,v1r,v2i,v2r
double precision:: q1,q2,q3,q4,q5,q6
integer:: i,k,kc,lvd2

!-----------------------------------------
 !Do k=0 first:
do i=0,nv-1
  t1r=a(i,2,0)+a(i,4,0)
  t2r=a(i,0,0)-0.5d0*t1r
  t3r=sinfpi3*(a(i,4,0)-a(i,2,0))
  u0r=a(i,0,0)+t1r
  t1i=a(i,5,0)+a(i,1,0)
  t2i=a(i,3,0)-0.5d0*t1i
  t3i=sinfpi3*(a(i,5,0)-a(i,1,0))
  v0r=a(i,3,0)+t1i
  b(i,0,0)=u0r+v0r
  b(i,0,1)=t2r-t2i
  b(i,0,2)=t2r+t2i
  b(i,0,3)=u0r-v0r
  b(i,0,4)=t3i-t3r
  b(i,0,5)=t3r+t3i
enddo
 !Next do remaining k:
if (nv .le. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      x1p=cosine(k,1)*a(i,1, k)-sine(k,1)*a(i,1,kc)
      y1p=cosine(k,1)*a(i,1,kc)+sine(k,1)*a(i,1, k)
      x2p=cosine(k,2)*a(i,2, k)-sine(k,2)*a(i,2,kc)
      y2p=cosine(k,2)*a(i,2,kc)+sine(k,2)*a(i,2, k)
      x3p=cosine(k,3)*a(i,3, k)-sine(k,3)*a(i,3,kc)
      y3p=cosine(k,3)*a(i,3,kc)+sine(k,3)*a(i,3, k)
      x4p=cosine(k,4)*a(i,4, k)-sine(k,4)*a(i,4,kc)
      y4p=cosine(k,4)*a(i,4,kc)+sine(k,4)*a(i,4, k)
      x5p=cosine(k,5)*a(i,5, k)-sine(k,5)*a(i,5,kc)
      y5p=cosine(k,5)*a(i,5,kc)+sine(k,5)*a(i,5, k)
      t1r=x2p+x4p
      t1i=y2p+y4p
      t2r=a(i,0,k)-0.5d0*t1r
      t2i=a(i,0,kc)-0.5d0*t1i
      t3r=sinfpi3*(x2p-x4p)
      t3i=sinfpi3*(y2p-y4p)
      u0r=a(i,0,k)+t1r
      u0i=a(i,0,kc)+t1i
      u1r=t2r+t3i
      u1i=t2i-t3r
      u2r=t2r-t3i
      u2i=t2i+t3r
      t1r=x5p+x1p
      t1i=y5p+y1p
      t2r=x3p-0.5d0*t1r
      t2i=y3p-0.5d0*t1i
      t3r=sinfpi3*(x5p-x1p)
      t3i=sinfpi3*(y5p-y1p)
      v0r=x3p+t1r
      v0i=y3p+t1i
      v1r=t2r+t3i
      v1i=t3r-t2i
      v2r=t2r-t3i
      v2i=t2i+t3r
      b(i, k,0)=u0r+v0r
      b(i,kc,0)=u2r-v2r
      b(i, k,1)=u1r-v1r
      b(i,kc,1)=u1r+v1r
      b(i, k,2)=u2r+v2r
      b(i,kc,2)=u0r-v0r
      b(i, k,3)=v0i-u0i
      b(i,kc,3)=u2i+v2i
      b(i, k,4)=v1i-u1i
      b(i,kc,4)=u1i+v1i
      b(i, k,5)=v2i-u2i
      b(i,kc,5)=u0i+v0i
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k,1)
    s1k=sine(k,1)
    c2k=cosine(k,2)
    s2k=sine(k,2)
    c3k=cosine(k,3)
    s3k=sine(k,3)
    c4k=cosine(k,4)
    s4k=sine(k,4)
    c5k=cosine(k,5)
    s5k=sine(k,5)
    do i=0,nv-1
      x1p=c1k*a(i,1, k)-s1k*a(i,1,kc)
      y1p=c1k*a(i,1,kc)+s1k*a(i,1, k)
      x2p=c2k*a(i,2, k)-s2k*a(i,2,kc)
      y2p=c2k*a(i,2,kc)+s2k*a(i,2, k)
      x3p=c3k*a(i,3, k)-s3k*a(i,3,kc)
      y3p=c3k*a(i,3,kc)+s3k*a(i,3, k)
      x4p=c4k*a(i,4, k)-s4k*a(i,4,kc)
      y4p=c4k*a(i,4,kc)+s4k*a(i,4, k)
      x5p=c5k*a(i,5, k)-s5k*a(i,5,kc)
      y5p=c5k*a(i,5,kc)+s5k*a(i,5, k)
      t1r=x2p+x4p
      t1i=y2p+y4p
      t2r=a(i,0,k)-0.5d0*t1r
      t2i=a(i,0,kc)-0.5d0*t1i
      t3r=sinfpi3*(x2p-x4p)
      t3i=sinfpi3*(y2p-y4p)
      u0r=a(i,0,k)+t1r
      u0i=a(i,0,kc)+t1i
      u1r=t2r+t3i
      u1i=t2i-t3r
      u2r=t2r-t3i
      u2i=t2i+t3r
      t1r=x5p+x1p
      t1i=y5p+y1p
      t2r=x3p-0.5d0*t1r
      t2i=y3p-0.5d0*t1i
      t3r=sinfpi3*(x5p-x1p)
      t3i=sinfpi3*(y5p-y1p)
      v0r=x3p+t1r
      v0i=y3p+t1i
      v1r=t2r+t3i
      v1i=t3r-t2i
      v2r=t2r-t3i
      v2i=t2i+t3r
      b(i, k,0)=u0r+v0r
      b(i,kc,0)=u2r-v2r
      b(i, k,1)=u1r-v1r
      b(i,kc,1)=u1r+v1r
      b(i, k,2)=u2r+v2r
      b(i,kc,2)=u0r-v0r
      b(i, k,3)=v0i-u0i
      b(i,kc,3)=u2i+v2i
      b(i, k,4)=v1i-u1i
      b(i,kc,4)=u1i+v1i
      b(i, k,5)=v2i-u2i
      b(i,kc,5)=u0i+v0i
    enddo
  enddo
endif

 !Catch the case k=lv/2 when lv even:
if (mod(lv,2) .eq. 0) then
  lvd2=lv/2
  do i=0,nv-1
    q1=a(i,2,lvd2)-a(i,4,lvd2)
    q2=a(i,0,lvd2)+0.5d0*q1
    q3=sinfpi3*(a(i,2,lvd2)+a(i,4,lvd2))
    q4=a(i,1,lvd2)+a(i,5,lvd2)
    q5=-a(i,3,lvd2)-0.5d0*q4
    q6=sinfpi3*(a(i,1,lvd2)-a(i,5,lvd2))
    b(i,lvd2,0)=q2+q6
    b(i,lvd2,1)=a(i,0,lvd2)-q1
    b(i,lvd2,2)=q2-q6
    b(i,lvd2,3)=q5+q3
    b(i,lvd2,4)=a(i,3,lvd2)-q4
    b(i,lvd2,5)=q5-q3
  enddo
endif
  
return
end subroutine
!================================================
subroutine forrdx5(a,b,nv,lv,cosine,sine)

! Radix five physical to Hermitian FFT with 'decimation in time'.

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:4,0:lv-1),b(0:nv-1,0:lv-1,0:4),cosine(0:lv-1,1:4),sine(0:lv-1,1:4)
 !Local declarations:
double precision,parameter:: rtf516=0.5590169943749474241022934171828190588601545899028814310677243113526302d0
double precision,parameter:: sinf2pi5=0.9510565162951535721164393333793821434056986341257502224473056444301532d0
double precision,parameter:: sinfpi5=0.5877852522924731291687059546390727685976524376431459910722724807572785d0
double precision,parameter:: sinrat=0.6180339887498948482045868343656381177203091798057628621354486227052605d0
double precision:: x1p,x2p,x3p,x4p,y1p,y2p,y3p,y4p
double precision:: s1k,s2k,s3k,s4k,c1k,c2k,c3k,c4k
double precision:: t1i,t1r,t2i,t2r,t3i,t3r,t4i,t4r,t5i,t5r,t6i,t6r
double precision:: t7i,t7r,t8i,t8r,t9i,t9r,t10i,t10r,t11i,t11r
integer:: i,k,kc

!--------------------------------------------------
 !Do k=0 first:
do i=0,nv-1
  t1r=a(i,1,0)+a(i,4,0)
  t2r=a(i,2,0)+a(i,3,0)
  t3r=sinf2pi5*(a(i,4,0)-a(i,1,0))
  t4r=sinf2pi5*(a(i,2,0)-a(i,3,0))
  t5r=t1r+t2r
  t6r=rtf516*(t1r-t2r)
  t7r=a(i,0,0)-0.25d0*t5r
  b(i,0,0)=a(i,0,0)+t5r
  b(i,0,1)=t7r+t6r
  b(i,0,2)=t7r-t6r
  b(i,0,3)=t4r+sinrat*t3r
  b(i,0,4)=t3r-sinrat*t4r
enddo
 !Next do remaining k:
if (nv .le. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      x1p=cosine(k,1)*a(i,1, k)-sine(k,1)*a(i,1,kc)
      y1p=cosine(k,1)*a(i,1,kc)+sine(k,1)*a(i,1, k)
      x2p=cosine(k,2)*a(i,2, k)-sine(k,2)*a(i,2,kc)
      y2p=cosine(k,2)*a(i,2,kc)+sine(k,2)*a(i,2, k)
      x3p=cosine(k,3)*a(i,3, k)-sine(k,3)*a(i,3,kc)
      y3p=cosine(k,3)*a(i,3,kc)+sine(k,3)*a(i,3, k)
      x4p=cosine(k,4)*a(i,4, k)-sine(k,4)*a(i,4,kc)
      y4p=cosine(k,4)*a(i,4,kc)+sine(k,4)*a(i,4, k)
      t1r=x1p+x4p
      t1i=y1p+y4p
      t2r=x2p+x3p
      t2i=y2p+y3p
      t3r=sinf2pi5*(x1p-x4p)
      t3i=sinf2pi5*(y1p-y4p)
      t4r=sinf2pi5*(x2p-x3p)
      t4i=sinf2pi5*(y2p-y3p)
      t5r=t1r+t2r
      t5i=t1i+t2i
      t6r=rtf516*(t1r-t2r)
      t6i=rtf516*(t1i-t2i)
      t7r=a(i,0,k)-0.25d0*t5r
      t7i=a(i,0,kc)-0.25d0*t5i
      t8r=t7r+t6r
      t8i=t7i+t6i
      t9r=t7r-t6r
      t9i=t7i-t6i
      t10r=t3r+sinrat*t4r
      t10i=t3i+sinrat*t4i
      t11r=t4r-sinrat*t3r
      t11i=sinrat*t3i-t4i
      b(i, k,0)=a(i,0,k)+t5r
      b(i,kc,0)=t8r-t10i
      b(i, k,1)=t8r+t10i
      b(i,kc,1)=t9r-t11i
      b(i, k,2)=t9r+t11i
      b(i,kc,2)=t9i+t11r
      b(i, k,3)=t11r-t9i
      b(i,kc,3)=t8i-t10r
      b(i, k,4)=-t8i-t10r
      b(i,kc,4)=a(i,0,kc)+t5i
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k,1)
    s1k=sine(k,1)
    c2k=cosine(k,2)
    s2k=sine(k,2)
    c3k=cosine(k,3)
    s3k=sine(k,3)
    c4k=cosine(k,4)
    s4k=sine(k,4)
    do i=0,nv-1
      x1p=c1k*a(i,1, k)-s1k*a(i,1,kc)
      y1p=c1k*a(i,1,kc)+s1k*a(i,1, k)
      x2p=c2k*a(i,2, k)-s2k*a(i,2,kc)
      y2p=c2k*a(i,2,kc)+s2k*a(i,2, k)
      x3p=c3k*a(i,3, k)-s3k*a(i,3,kc)
      y3p=c3k*a(i,3,kc)+s3k*a(i,3, k)
      x4p=c4k*a(i,4, k)-s4k*a(i,4,kc)
      y4p=c4k*a(i,4,kc)+s4k*a(i,4, k)
      t1r=x1p+x4p
      t1i=y1p+y4p
      t2r=x2p+x3p
      t2i=y2p+y3p
      t3r=sinf2pi5*(x1p-x4p)
      t3i=sinf2pi5*(y1p-y4p)
      t4r=sinf2pi5*(x2p-x3p)
      t4i=sinf2pi5*(y2p-y3p)
      t5r=t1r+t2r
      t5i=t1i+t2i
      t6r=rtf516*(t1r-t2r)
      t6i=rtf516*(t1i-t2i)
      t7r=a(i,0,k)-0.25d0*t5r
      t7i=a(i,0,kc)-0.25d0*t5i
      t8r=t7r+t6r
      t8i=t7i+t6i
      t9r=t7r-t6r
      t9i=t7i-t6i
      t10r=t3r+sinrat*t4r
      t10i=t3i+sinrat*t4i
      t11r=t4r-sinrat*t3r
      t11i=sinrat*t3i-t4i
      b(i, k,0)=a(i,0,k)+t5r
      b(i,kc,0)=t8r-t10i
      b(i, k,1)=t8r+t10i
      b(i,kc,1)=t9r-t11i
      b(i, k,2)=t9r+t11i
      b(i,kc,2)=t9i+t11r
      b(i, k,3)=t11r-t9i
      b(i,kc,3)=t8i-t10r
      b(i, k,4)=-t8i-t10r
      b(i,kc,4)=a(i,0,kc)+t5i
    enddo
  enddo
endif

return
end subroutine
!===========================================
subroutine forrdx4(a,b,nv,lv,cosine,sine)

! Radix four physical to Hermitian FFT with 'decimation in time'.

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:3,0:lv-1),b(0:nv-1,0:lv-1,0:3),cosine(0:lv-1,1:3),sine(0:lv-1,1:3)

 !Local declarations:
double precision,parameter:: rtf12=0.7071067811865475244008443621048490392848359376884740365883398689953662d0
double precision:: x1p,x2p,x3p,y1p,y2p,y3p
double precision:: s1k,s2k,s3k,c1k,c2k,c3k
double precision:: t1i,t1r,t2i,t2r,t3i,t3r,t4i,t4r
double precision:: q1,q2
integer:: i,k,kc,lvd2

!-----------------------------------------------
 !Do k=0 first:
do i=0,nv-1
  t1r=a(i,0,0)+a(i,2,0)
  t2r=a(i,1,0)+a(i,3,0)
  b(i,0,0)=t1r+t2r
  b(i,0,1)=a(i,0,0)-a(i,2,0)
  b(i,0,2)=t1r-t2r
  b(i,0,3)=a(i,3,0)-a(i,1,0)
enddo
 !Next do remaining k:
if (nv .lt. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      x1p=cosine(k,1)*a(i,1, k)-sine(k,1)*a(i,1,kc)
      y1p=cosine(k,1)*a(i,1,kc)+sine(k,1)*a(i,1, k)
      x2p=cosine(k,2)*a(i,2, k)-sine(k,2)*a(i,2,kc)
      y2p=cosine(k,2)*a(i,2,kc)+sine(k,2)*a(i,2, k)
      x3p=cosine(k,3)*a(i,3, k)-sine(k,3)*a(i,3,kc)
      y3p=cosine(k,3)*a(i,3,kc)+sine(k,3)*a(i,3, k)
      t1r=a(i,0,k)+x2p
      t1i=a(i,0,kc)+y2p
      t2r=x1p+x3p
      t2i=y1p+y3p
      t3r=a(i,0,k)-x2p
      t3i=a(i,0,kc)-y2p
      t4r=x3p-x1p
      t4i=y1p-y3p
      b(i, k,0)=t1r+t2r
      b(i,kc,0)=t3r-t4i
      b(i, k,1)=t3r+t4i
      b(i,kc,1)=t1r-t2r
      b(i, k,2)=t2i-t1i
      b(i,kc,2)=t3i+t4r
      b(i, k,3)=t4r-t3i
      b(i,kc,3)=t1i+t2i
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k,1)
    s1k=sine(k,1)
    c2k=cosine(k,2)
    s2k=sine(k,2)
    c3k=cosine(k,3)
    s3k=sine(k,3)
    do i=0,nv-1
      x1p=c1k*a(i,1, k)-s1k*a(i,1,kc)
      y1p=c1k*a(i,1,kc)+s1k*a(i,1, k)
      x2p=c2k*a(i,2, k)-s2k*a(i,2,kc)
      y2p=c2k*a(i,2,kc)+s2k*a(i,2, k)
      x3p=c3k*a(i,3, k)-s3k*a(i,3,kc)
      y3p=c3k*a(i,3,kc)+s3k*a(i,3, k)
      t1r=a(i,0,k)+x2p
      t1i=a(i,0,kc)+y2p
      t2r=x1p+x3p
      t2i=y1p+y3p
      t3r=a(i,0,k)-x2p
      t3i=a(i,0,kc)-y2p
      t4r=x3p-x1p
      t4i=y1p-y3p
      b(i, k,0)=t1r+t2r
      b(i,kc,0)=t3r-t4i
      b(i, k,1)=t3r+t4i
      b(i,kc,1)=t1r-t2r
      b(i, k,2)=t2i-t1i
      b(i,kc,2)=t3i+t4r
      b(i, k,3)=t4r-t3i
      b(i,kc,3)=t1i+t2i
    enddo
  enddo
endif

 !Catch the case k=lv/2 when lv even:
if (mod(lv,2) .eq. 0) then
  lvd2=lv/2
  do i=0,nv-1
    q1=rtf12*(a(i,1,lvd2)-a(i,3,lvd2))
    q2=rtf12*(a(i,1,lvd2)+a(i,3,lvd2))
    b(i,lvd2,0)=a(i,0,lvd2)+q1
    b(i,lvd2,1)=a(i,0,lvd2)-q1
    b(i,lvd2,2)=a(i,2,lvd2)-q2
    b(i,lvd2,3)=-a(i,2,lvd2)-q2
  enddo
endif

return
end subroutine 
!================================================

subroutine forrdx3(a,b,nv,lv,cosine,sine)

! Radix three physical to Hermitian FFT with 'decimation in time'.

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:2,0:lv-1),b(0:nv-1,0:lv-1,0:2),cosine(0:lv-1,1:2),sine(0:lv-1,1:2)

 !Local declarations:
double precision,parameter:: sinfpi3=0.8660254037844386467637231707529361834714026269051903140279034897259665d0
double precision:: x1p,x2p,y1p,y2p
double precision:: s1k,s2k,c1k,c2k
double precision:: t1i,t1r,t2i,t2r,t3i,t3r
integer:: i,k,kc

!---------------------------------------------
 !Do k=0 first:
do i=0,nv-1
  t1r=a(i,1,0)+a(i,2,0)
  b(i,0,0)=a(i,0,0)+t1r
  b(i,0,1)=a(i,0,0)-0.5d0*t1r
  b(i,0,2)=sinfpi3*(a(i,2,0)-a(i,1,0))
enddo
 !Next do remaining k:
if (nv .le. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      x1p=cosine(k,1)*a(i,1, k)-sine(k,1)*a(i,1,kc)
      y1p=cosine(k,1)*a(i,1,kc)+sine(k,1)*a(i,1, k)
      x2p=cosine(k,2)*a(i,2, k)-sine(k,2)*a(i,2,kc)
      y2p=cosine(k,2)*a(i,2,kc)+sine(k,2)*a(i,2, k)
      t1r=x1p+x2p
      t1i=y1p+y2p
      t2r=a(i,0, k)-0.5d0*t1r
      t2i=0.5d0*t1i-a(i,0,kc)
      t3r=sinfpi3*(x2p-x1p)
      t3i=sinfpi3*(y1p-y2p)
      b(i, k,0)=a(i,0, k)+t1r
      b(i,kc,0)=t2r-t3i
      b(i, k,1)=t2r+t3i
      b(i,kc,1)=t3r-t2i
      b(i, k,2)=t2i+t3r
      b(i,kc,2)=a(i,0,kc)+t1i
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k,1)
    s1k=sine(k,1)
    c2k=cosine(k,2)
    s2k=sine(k,2)
    do i=0,nv-1
      x1p=c1k*a(i,1, k)-s1k*a(i,1,kc)
      y1p=c1k*a(i,1,kc)+s1k*a(i,1, k)
      x2p=c2k*a(i,2, k)-s2k*a(i,2,kc)
      y2p=c2k*a(i,2,kc)+s2k*a(i,2, k)
      t1r=x1p+x2p
      t1i=y1p+y2p
      t2r=a(i,0, k)-0.5d0*t1r
      t2i=0.5d0*t1i-a(i,0,kc)
      t3r=sinfpi3*(x2p-x1p)
      t3i=sinfpi3*(y1p-y2p)
      b(i, k,0)=a(i,0, k)+t1r
      b(i,kc,0)=t2r-t3i
      b(i, k,1)=t2r+t3i
      b(i,kc,1)=t3r-t2i
      b(i, k,2)=t2i+t3r
      b(i,kc,2)=a(i,0,kc)+t1i
    enddo
  enddo
endif

return
end subroutine
!========================================

subroutine forrdx2(a,b,nv,lv,cosine,sine)

! Radix two physical to Hermitian FFT with 'decimation in time'.

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:1,0:lv-1),b(0:nv-1,0:lv-1,0:1),cosine(0:lv-1),sine(0:lv-1)

 !Local declarations: 
double precision:: x1,y1,c1k,s1k
integer:: i,k,kc

!-----------------------------------------
 !Do k=0 first:
do i=0,nv-1
  b(i,0,0)=a(i,0,0)+a(i,1,0)
  b(i,0,1)=a(i,0,0)-a(i,1,0)
enddo
 !Next do remaining k:
if (nv .lt. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      x1=cosine(k)*a(i,1, k)-sine(k)*a(i,1,kc)
      y1=cosine(k)*a(i,1,kc)+sine(k)*a(i,1, k)
      b(i, k,0)=a(i,0, k)+x1
      b(i,kc,0)=a(i,0, k)-x1
      b(i, k,1)=y1-a(i,0,kc)
      b(i,kc,1)=a(i,0,kc)+y1
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k)
    s1k=sine(k)
    do i=0,nv-1
      x1=c1k*a(i,1, k)-s1k*a(i,1,kc)
      y1=c1k*a(i,1,kc)+s1k*a(i,1, k)
      b(i, k,0)=a(i,0, k)+x1
      b(i,kc,0)=a(i,0, k)-x1
      b(i, k,1)=y1-a(i,0,kc)
      b(i,kc,1)=a(i,0,kc)+y1
    enddo
  enddo
endif

return
end subroutine
!======================================

!====================================================	
! Spectral to physical (reverse) routines:
!====================================================	
subroutine revrdx6(a,b,nv,lv,cosine,sine)

!Radix six Hermitian to physical FFT with 'decimation in frequency'.      

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:lv-1,0:5),b(0:nv-1,0:5,0:lv-1),cosine(0:lv-1,1:5),sine(0:lv-1,1:5)

 !Local declarations:
double precision,parameter:: sinfpi3=0.8660254037844386467637231707529361834714026269051903140279034897259665d0
double precision:: x1p,x2p,x3p,x4p,x5p
double precision:: y1p,y2p,y3p,y4p,y5p
double precision:: s1k,s2k,s3k,s4k,s5k
double precision:: c1k,c2k,c3k,c4k,c5k
double precision:: t1i,t1r,t2i,t2r,t3i,t3r
double precision:: u0i,u0r,u1i,u1r,u2i,u2r
double precision:: v0i,v0r,v1i,v1r,v2i,v2r
double precision:: q1,q2,q3,q4,q5,q6
integer:: i,k,kc,lvd2

!-----------------------------------------
 !Do k=0 first:
do i=0,nv-1
  t2r=a(i,0,0)-0.5d0*a(i,0,2)
  t3r=sinfpi3*a(i,0,4)
  u0r=a(i,0,0)+a(i,0,2)
  u1r=t2r+t3r
  u2r=t2r-t3r
  t2i=a(i,0,3)-0.5d0*a(i,0,1)
  t3i=-sinfpi3*a(i,0,5)
  v0r=a(i,0,3)+a(i,0,1)
  v1r=t2i+t3i
  v2r=t2i-t3i
  b(i,0,0)=u0r+v0r
  b(i,1,0)=u1r-v1r
  b(i,2,0)=u2r+v2r
  b(i,3,0)=u0r-v0r
  b(i,4,0)=u1r+v1r
  b(i,5,0)=u2r-v2r
enddo
 !Next do remaining k:
if (nv .le. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      t1r=a(i, k,2)+a(i,kc,1)
      t1i=a(i,kc,3)-a(i, k,4)
      t2r=a(i, k,0)-0.5d0*t1r
      t2i=a(i,kc,5)-0.5d0*t1i
      t3r=sinfpi3*(a(i, k,2)-a(i,kc,1))
      t3i=sinfpi3*(a(i,kc,3)+a(i, k,4))
      u0r=a(i, k,0)+t1r
      u0i=a(i,kc,5)+t1i
      u1r=t2r+t3i
      u1i=t2i-t3r
      u2r=t2r-t3i
      u2i=t2i+t3r
      t1r=a(i,kc,0)+a(i,k,1)
      t1i=a(i,kc,4)-a(i,k,5)
      t2r=a(i,kc,2)-0.5d0*t1r
      t2i=-a(i,k,3)-0.5d0*t1i
      t3r=sinfpi3*(a(i,kc,0)-a(i, k,1))
      t3i=sinfpi3*(-a(i,k,5)-a(i,kc,4))
      v0r=a(i,kc,2)+t1r
      v0i=t1i-a(i,k,3)
      v1r=t2r+t3i
      v1i=t2i-t3r
      v2r=t2r-t3i
      v2i=t2i+t3r
      x1p=u1r-v1r
      y1p=u1i-v1i
      x2p=u2r+v2r
      y2p=u2i+v2i
      x3p=u0r-v0r
      y3p=u0i-v0i
      x4p=u1r+v1r
      y4p=u1i+v1i
      x5p=u2r-v2r
      y5p=u2i-v2i
      b(i,0, k)=u0r+v0r
      b(i,0,kc)=u0i+v0i
      b(i,1, k)=cosine(k,1)*x1p-sine(k,1)*y1p
      b(i,1,kc)=cosine(k,1)*y1p+sine(k,1)*x1p
      b(i,2, k)=cosine(k,2)*x2p-sine(k,2)*y2p
      b(i,2,kc)=cosine(k,2)*y2p+sine(k,2)*x2p
      b(i,3, k)=cosine(k,3)*x3p-sine(k,3)*y3p
      b(i,3,kc)=cosine(k,3)*y3p+sine(k,3)*x3p
      b(i,4, k)=cosine(k,4)*x4p-sine(k,4)*y4p
      b(i,4,kc)=cosine(k,4)*y4p+sine(k,4)*x4p
      b(i,5, k)=cosine(k,5)*x5p-sine(k,5)*y5p
      b(i,5,kc)=cosine(k,5)*y5p+sine(k,5)*x5p
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k,1)
    s1k=sine(k,1)
    c2k=cosine(k,2)
    s2k=sine(k,2)
    c3k=cosine(k,3)
    s3k=sine(k,3)
    c4k=cosine(k,4)
    s4k=sine(k,4)
    c5k=cosine(k,5)
    s5k=sine(k,5)
    do i=0,nv-1
      t1r=a(i, k,2)+a(i,kc,1)
      t1i=a(i,kc,3)-a(i, k,4)
      t2r=a(i, k,0)-0.5d0*t1r
      t2i=a(i,kc,5)-0.5d0*t1i
      t3r=sinfpi3*(a(i, k,2)-a(i,kc,1))
      t3i=sinfpi3*(a(i,kc,3)+a(i, k,4))
      u0r=a(i, k,0)+t1r
      u0i=a(i,kc,5)+t1i
      u1r=t2r+t3i
      u1i=t2i-t3r
      u2r=t2r-t3i
      u2i=t2i+t3r
      t1r=a(i,kc,0)+a(i,k,1)
      t1i=a(i,kc,4)-a(i,k,5)
      t2r=a(i,kc,2)-0.5d0*t1r
      t2i=-a(i,k,3)-0.5d0*t1i
      t3r=sinfpi3*(a(i,kc,0)-a(i, k,1))
      t3i=sinfpi3*(-a(i,k,5)-a(i,kc,4))
      v0r=a(i,kc,2)+t1r
      v0i=t1i-a(i,k,3)
      v1r=t2r+t3i
      v1i=t2i-t3r
      v2r=t2r-t3i
      v2i=t2i+t3r
      x1p=u1r-v1r
      y1p=u1i-v1i
      x2p=u2r+v2r
      y2p=u2i+v2i
      x3p=u0r-v0r
      y3p=u0i-v0i
      x4p=u1r+v1r
      y4p=u1i+v1i
      x5p=u2r-v2r
      y5p=u2i-v2i
      b(i,0, k)=u0r+v0r
      b(i,0,kc)=u0i+v0i
      b(i,1, k)=c1k*x1p-s1k*y1p
      b(i,1,kc)=c1k*y1p+s1k*x1p
      b(i,2, k)=c2k*x2p-s2k*y2p
      b(i,2,kc)=c2k*y2p+s2k*x2p
      b(i,3, k)=c3k*x3p-s3k*y3p
      b(i,3,kc)=c3k*y3p+s3k*x3p
      b(i,4, k)=c4k*x4p-s4k*y4p
      b(i,4,kc)=c4k*y4p+s4k*x4p
      b(i,5, k)=c5k*x5p-s5k*y5p
      b(i,5,kc)=c5k*y5p+s5k*x5p
    enddo
  enddo
endif

 !Catch the case k=lv/2 when lv even:
if (mod(lv,2) .eq. 0) then
  lvd2=lv/2
  do i=0,nv-1
    q1=a(i,lvd2,0)+a(i,lvd2,2)
    q2=a(i,lvd2,5)+a(i,lvd2,3)
    q3=a(i,lvd2,1)-0.5d0*q1
    q4=a(i,lvd2,4)+0.5d0*q2
    q5=sinfpi3*(a(i,lvd2,0)-a(i,lvd2,2))
    q6=sinfpi3*(a(i,lvd2,5)-a(i,lvd2,3))
    b(i,0,lvd2)=a(i,lvd2,1)+q1
    b(i,1,lvd2)=q4+q5
    b(i,2,lvd2)=q6-q3
    b(i,3,lvd2)=q2-a(i,lvd2,4)
    b(i,4,lvd2)=q3+q6
    b(i,5,lvd2)=q4-q5
  enddo
endif

return
end subroutine
!=======================================

subroutine revrdx5(a,b,nv,lv,cosine,sine)

! Radix five Hermitian to physical FFT with 'decimation in frequency'.

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:lv-1,0:4),b(0:nv-1,0:4,0:lv-1),cosine(0:lv-1,1:4),sine(0:lv-1,1:4)
 !Local declarations: 
double precision,parameter:: rtf516=0.5590169943749474241022934171828190588601545899028814310677243113526302d0
double precision,parameter:: sinf2pi5=0.9510565162951535721164393333793821434056986341257502224473056444301532d0
double precision,parameter:: sinfpi5=0.5877852522924731291687059546390727685976524376431459910722724807572785d0
double precision,parameter:: sinrat=0.6180339887498948482045868343656381177203091798057628621354486227052605d0
double precision:: x1p,x2p,x3p,x4p,y1p,y2p,y3p,y4p
double precision:: s1k,s2k,s3k,s4k,c1k,c2k,c3k,c4k
double precision:: t1i,t1r,t2i,t2r,t3i,t3r,t4i,t4r,t5i,t5r,t6i,t6r
double precision:: t7i,t7r,t8i,t8r,t9i,t9r,t10i,t10r,t11i,t11r
integer:: i,k,kc

!----------------------------------------------------
 !Do k=0 first:
do i=0,nv-1
  t3r=sinf2pi5*a(i,0,4)
  t4r=sinf2pi5*a(i,0,3)
  t5r=a(i,0,1)+a(i,0,2)
  t6r=rtf516*(a(i,0,1)-a(i,0,2))
  t7r=a(i,0,0)-0.25d0*t5r
  t8r=t7r+t6r
  t9r=t7r-t6r
  t10r=t3r+sinrat*t4r
  t11r=sinrat*t3r-t4r
  b(i,0,0)=a(i,0,0)+t5r
  b(i,1,0)=t8r+t10r
  b(i,2,0)=t9r+t11r
  b(i,3,0)=t9r-t11r
  b(i,4,0)=t8r-t10r
enddo
 !Next do remaining k:
if (nv .le. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      t1r=a(i, k,1)+a(i,kc,0)
      t1i=a(i,kc,3)-a(i, k,4)
      t2r=a(i, k,2)+a(i,kc,1)
      t2i=a(i,kc,2)-a(i, k,3)
      t3r=sinf2pi5*(a(i, k,1)-a(i,kc,0))
      t3i=sinf2pi5*(a(i,kc,3)+a(i, k,4))
      t4r=sinf2pi5*(a(i, k,2)-a(i,kc,1))
      t4i=sinf2pi5*(a(i,kc,2)+a(i, k,3))
      t5r=t1r+t2r
      t5i=t1i+t2i
      t6r=rtf516*(t1r-t2r)
      t6i=rtf516*(t1i-t2i)
      t7r=a(i,k,0)-0.25d0*t5r
      t7i=a(i,kc,4)-0.25d0*t5i
      t8r=t7r+t6r
      t8i=t7i+t6i
      t9r=t7r-t6r
      t9i=t7i-t6i
      t10r=t3r+sinrat*t4r
      t10i=t3i+sinrat*t4i
      t11r=sinrat*t3r-t4r
      t11i=sinrat*t3i-t4i
      x1p=t8r+t10i
      y1p=t8i-t10r
      x2p=t9r+t11i
      y2p=t9i-t11r
      x3p=t9r-t11i
      y3p=t9i+t11r
      x4p=t8r-t10i
      y4p=t8i+t10r
      b(i,0, k)=a(i, k,0)+t5r
      b(i,0,kc)=a(i,kc,4)+t5i
      b(i,1, k)=cosine(k,1)*x1p-sine(k,1)*y1p
      b(i,1,kc)=cosine(k,1)*y1p+sine(k,1)*x1p
      b(i,2, k)=cosine(k,2)*x2p-sine(k,2)*y2p
      b(i,2,kc)=cosine(k,2)*y2p+sine(k,2)*x2p
      b(i,3, k)=cosine(k,3)*x3p-sine(k,3)*y3p
      b(i,3,kc)=cosine(k,3)*y3p+sine(k,3)*x3p
      b(i,4, k)=cosine(k,4)*x4p-sine(k,4)*y4p
      b(i,4,kc)=cosine(k,4)*y4p+sine(k,4)*x4p
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k,1)
    s1k=sine(k,1)
    c2k=cosine(k,2)
    s2k=sine(k,2)
    c3k=cosine(k,3)
    s3k=sine(k,3)
    c4k=cosine(k,4)
    s4k=sine(k,4)
    do i=0,nv-1
      t1r=a(i, k,1)+a(i,kc,0)
      t1i=a(i,kc,3)-a(i, k,4)
      t2r=a(i, k,2)+a(i,kc,1)
      t2i=a(i,kc,2)-a(i, k,3)
      t3r=sinf2pi5*(a(i, k,1)-a(i,kc,0))
      t3i=sinf2pi5*(a(i,kc,3)+a(i, k,4))
      t4r=sinf2pi5*(a(i, k,2)-a(i,kc,1))
      t4i=sinf2pi5*(a(i,kc,2)+a(i, k,3))
      t5r=t1r+t2r
      t5i=t1i+t2i
      t6r=rtf516*(t1r-t2r)
      t6i=rtf516*(t1i-t2i)
      t7r=a(i,k,0)-0.25d0*t5r
      t7i=a(i,kc,4)-0.25d0*t5i
      t8r=t7r+t6r
      t8i=t7i+t6i
      t9r=t7r-t6r
      t9i=t7i-t6i
      t10r=t3r+sinrat*t4r
      t10i=t3i+sinrat*t4i
      t11r=sinrat*t3r-t4r
      t11i=sinrat*t3i-t4i
      x1p=t8r+t10i
      y1p=t8i-t10r
      x2p=t9r+t11i
      y2p=t9i-t11r
      x3p=t9r-t11i
      y3p=t9i+t11r
      x4p=t8r-t10i
      y4p=t8i+t10r
      b(i,0, k)=a(i, k,0)+t5r
      b(i,0,kc)=a(i,kc,4)+t5i
      b(i,1, k)=c1k*x1p-s1k*y1p
      b(i,1,kc)=c1k*y1p+s1k*x1p
      b(i,2, k)=c2k*x2p-s2k*y2p
      b(i,2,kc)=c2k*y2p+s2k*x2p
      b(i,3, k)=c3k*x3p-s3k*y3p
      b(i,3,kc)=c3k*y3p+s3k*x3p
      b(i,4, k)=c4k*x4p-s4k*y4p
      b(i,4,kc)=c4k*y4p+s4k*x4p
    enddo
  enddo
endif

return
end subroutine 
!=================================================

subroutine revrdx4(a,b,nv,lv,cosine,sine)

!Radix four Hermitian to physical FFT with 'decimation in frequency'.      

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:lv-1,0:3),b(0:nv-1,0:3,0:lv-1),cosine(0:lv-1,1:3),sine(0:lv-1,1:3)

 !Local declarations:
double precision,parameter:: rtf12=0.7071067811865475244008443621048490392848359376884740365883398689953662d0
double precision:: x1p,x2p,x3p,y1p,y2p,y3p
double precision:: s1k,s2k,s3k,c1k,c2k,c3k
double precision:: t1i,t1r,t2i,t2r,t3i,t3r,t4i,t4r
integer:: i,k,kc,lvd2

!--------------------------------------------------
 !Do k=0 first:
do i=0,nv-1
  t1r=a(i,0,0)+a(i,0,2)
  t2r=a(i,0,1)
  t3r=a(i,0,0)-a(i,0,2)
  t4r=a(i,0,3)
  b(i,0,0)=t1r+t2r
  b(i,1,0)=t3r+t4r
  b(i,2,0)=t1r-t2r
  b(i,3,0)=t3r-t4r
enddo
 !Next do remaining k:
if (nv .lt. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      t1r=a(i, k,0)+a(i,kc,1)
      t1i=a(i,kc,3)-a(i, k,2)
      t2r=a(i, k,1)+a(i,kc,0)
      t2i=a(i,kc,2)-a(i, k,3)
      t3r=a(i, k,0)-a(i,kc,1)
      t3i=a(i,kc,3)+a(i, k,2)
      t4r=a(i, k,1)-a(i,kc,0)
      t4i=a(i,kc,2)+a(i, k,3)
      x1p=t3r+t4i
      y1p=t3i-t4r
      x2p=t1r-t2r
      y2p=t1i-t2i
      x3p=t3r-t4i
      y3p=t3i+t4r
      b(i,0, k)=t1r+t2r
      b(i,0,kc)=t1i+t2i
      b(i,1, k)=cosine(k,1)*x1p-sine(k,1)*y1p
      b(i,1,kc)=cosine(k,1)*y1p+sine(k,1)*x1p
      b(i,2, k)=cosine(k,2)*x2p-sine(k,2)*y2p
      b(i,2,kc)=cosine(k,2)*y2p+sine(k,2)*x2p
      b(i,3, k)=cosine(k,3)*x3p-sine(k,3)*y3p
      b(i,3,kc)=cosine(k,3)*y3p+sine(k,3)*x3p
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k,1)
    s1k=sine(k,1)
    c2k=cosine(k,2)
    s2k=sine(k,2)
    c3k=cosine(k,3)
    s3k=sine(k,3)
    do i=0,nv-1
      t1r=a(i,k,0)+a(i,kc,1)
      t1i=a(i,kc,3)-a(i,k,2)
      t2r=a(i,k,1)+a(i,kc,0)
      t2i=a(i,kc,2)-a(i,k,3)
      t3r=a(i,k,0)-a(i,kc,1)
      t3i=a(i,kc,3)+a(i,k,2)
      t4r=a(i,k,1)-a(i,kc,0)
      t4i=a(i,kc,2)+a(i,k,3)
      x1p=t3r+t4i
      y1p=t3i-t4r
      x2p=t1r-t2r
      y2p=t1i-t2i
      x3p=t3r-t4i
      y3p=t3i+t4r
      b(i,0, k)=t1r+t2r
      b(i,0,kc)=t1i+t2i
      b(i,1, k)=c1k*x1p-s1k*y1p
      b(i,1,kc)=c1k*y1p+s1k*x1p
      b(i,2, k)=c2k*x2p-s2k*y2p
      b(i,2,kc)=c2k*y2p+s2k*x2p
      b(i,3, k)=c3k*x3p-s3k*y3p
      b(i,3,kc)=c3k*y3p+s3k*x3p
    enddo
  enddo
endif

 !Catch the case k=lv/2 when lv even:
if (mod(lv,2) .eq. 0) then
  lvd2=lv/2
  do i=0,nv-1
    b(i,0,lvd2)=a(i,lvd2,0)+a(i,lvd2,1)
    b(i,2,lvd2)=a(i,lvd2,3)-a(i,lvd2,2)
    t3r=a(i,lvd2,0)-a(i,lvd2,1)
    t4r=a(i,lvd2,3)+a(i,lvd2,2)
    b(i,1,lvd2)=rtf12*(t3r+t4r)
    b(i,3,lvd2)=rtf12*(t4r-t3r)
  enddo
endif

return
end subroutine
!=================================================

subroutine revrdx3(a,b,nv,lv,cosine,sine)

!Radix three Hermitian to physical FFT with 'decimation in frequency'.      

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:lv-1,0:2),b(0:nv-1,0:2,0:lv-1),cosine(0:lv-1,1:2),sine(0:lv-1,1:2)
 
 !Local declarations:
double precision,parameter:: sinfpi3=0.8660254037844386467637231707529361834714026269051903140279034897259665d0
double precision:: x1p,x2p,y1p,y2p
double precision:: c2k,c1k,s2k,s1k
double precision:: t1i,t1r,t2i,t2r,t3i,t3r
integer:: i,k,kc

!-------------------------------------------------
 !Do k=0 first:
do i=0,nv-1
  t1r=a(i,0,1)
  t2r=a(i,0,0)-0.5d0*t1r
  t3r=sinfpi3*a(i,0,2)
  b(i,0,0)=a(i,0,0)+t1r
  b(i,1,0)=t2r+t3r
  b(i,2,0)=t2r-t3r
enddo
 !Next do remaining k:
if (nv .le. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      t1r=a(i, k,1)+a(i,kc,0)
      t1i=a(i,kc,1)-a(i, k,2)
      t2r=a(i, k,0)-0.5d0*t1r
      t2i=a(i,kc,2)-0.5d0*t1i
      t3r=sinfpi3*(a(i, k,1)-a(i,kc,0))
      t3i=sinfpi3*(a(i,kc,1)+a(i, k,2))
      x1p=t2r+t3i
      y1p=t2i-t3r
      x2p=t2r-t3i
      y2p=t2i+t3r
      b(i,0, k)=a(i, k,0)+t1r
      b(i,0,kc)=a(i,kc,2)+t1i
      b(i,1, k)=cosine(k,1)*x1p-sine(k,1)*y1p
      b(i,1,kc)=sine(k,1)*x1p+cosine(k,1)*y1p
      b(i,2, k)=cosine(k,2)*x2p-sine(k,2)*y2p
      b(i,2,kc)=sine(k,2)*x2p+cosine(k,2)*y2p
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k,1)
    s1k=sine(k,1)
    c2k=cosine(k,2)
    s2k=sine(k,2)
    do i=0,nv-1
      t1r=a(i, k,1)+a(i,kc,0)
      t1i=a(i,kc,1)-a(i, k,2)
      t2r=a(i, k,0)-0.5d0*t1r
      t2i=a(i,kc,2)-0.5d0*t1i
      t3r=sinfpi3*(a(i, k,1)-a(i,kc,0))
      t3i=sinfpi3*(a(i,kc,1)+a(i, k,2))
      x1p=t2r+t3i
      y1p=t2i-t3r
      x2p=t2r-t3i
      y2p=t2i+t3r
      b(i,0, k)=a(i, k,0)+t1r
      b(i,0,kc)=a(i,kc,2)+t1i
      b(i,1, k)=c1k*x1p-s1k*y1p
      b(i,1,kc)=s1k*x1p+c1k*y1p
      b(i,2, k)=c2k*x2p-s2k*y2p
      b(i,2,kc)=s2k*x2p+c2k*y2p
    enddo
  enddo
endif

return
end subroutine
!================================================

subroutine revrdx2(a,b,nv,lv,cosine,sine)

!Radix two Hermitian to physical FFT with 'decimation in frequency'.      

implicit none

 !Arguments declarations:
integer:: nv,lv
double precision:: a(0:nv-1,0:lv-1,0:1),b(0:nv-1,0:1,0:lv-1),cosine(0:lv-1),sine(0:lv-1)

 !Local declarations:
double precision:: x1p,y1p,c1k,s1k
integer:: i,k,kc

!-----------------------------------------
 !Do k=0 first:
do i=0,nv-1
  b(i,0,0)=a(i,0,0)+a(i,0,1)
  b(i,1,0)=a(i,0,0)-a(i,0,1)
enddo
 !Next do remaining k:
if (nv .lt. (lv-1)/2) then
  do i=0,nv-1
    do k=1,(lv-1)/2
      kc=lv-k
      x1p=a(i, k,0)-a(i,kc,0)
      y1p=a(i,kc,1)+a(i, k,1)
      b(i,0, k)=a(i, k,0)+a(i,kc,0)
      b(i,0,kc)=a(i,kc,1)-a(i, k,1)
      b(i,1, k)=cosine(k)*x1p-sine(k)*y1p
      b(i,1,kc)=cosine(k)*y1p+sine(k)*x1p
    enddo
  enddo
else
  do k=1,(lv-1)/2
    kc=lv-k
    c1k=cosine(k)
    s1k=sine(k)
    do i=0,nv-1
      x1p=a(i, k,0)-a(i,kc,0)
      y1p=a(i,kc,1)+a(i, k,1)
      b(i,0, k)=a(i, k,0)+a(i,kc,0)
      b(i,0,kc)=a(i,kc,1)-a(i, k,1)
      b(i,1, k)=c1k*x1p-s1k*y1p
      b(i,1,kc)=c1k*y1p+s1k*x1p
    enddo
  enddo
endif

return
end subroutine
!=================================================

end module
