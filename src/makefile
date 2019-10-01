 # Set up source and install directories,
 # these can be edited if the source/install structure needed is different to the standard.
sourcedir = .
installdir = ..

 # Set main source and object files:
sources = $(sourcedir)/parameters.f90 $(sourcedir)/constants.f90 $(sourcedir)/spectral.f90 #Main f90 sources
objects = parameters.o constants.o spectral.o #Main .o files 

 # Set location of fft library:
fft_lib = stafft.f90 sta2dfft.f90 

 # Set flag options: 
norm_flags = -O3 -mcmodel=medium #Compiler flags
test_flags = -O3 -mcmodel=medium -fbounds-check -fbacktrace  #Compiler flags

 # Set compiler and flag used:
f90=gfortran
flags=$(norm_flags)

#----------------------------------------------------------------------
 # Rules:
all: nhswps setup proxy_init_all proxy_post_all install

nhswps: $(objects) $(fft_lib) $(sourcedir)/nhswps.f90
	$(f90) $(fft_lib) $(objects) $(sourcedir)/nhswps.f90 -o nhswps $(flags)

$(objects): $(fft_lib) $(sources) 
	$(f90) $(fft_lib) $(sources) -c $(flags)

.PHONY: clean setup install
clean:
	rm -f *.o *.mod
setup:
	test -d $(installdir)/3d || mkdir $(installdir)/3d
	test -d $(installdir)/2d || mkdir $(installdir)/2d

install:
	find -type f -perm /a+x -exec /bin/mv {} $(installdir)/ \;

 # Include intialisation and post-processing rules 
 # if these exist:
-include init/init.make
-include post/post.make

 # Proxy rules to ensure whatever exists in the sub-directories
 # also gets built:
ifdef init_exists
proxy_init_all: init_all
else 
proxy_init_all:
	 @echo No init directory found 
endif

ifdef post_exists
proxy_post_all: post_all
else 
proxy_post_all:
	 @echo No post directory found 
endif
