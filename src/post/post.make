 # Set existence of directory variable used in main makefile:
post_exists = true
 # Calculate f90 codes existing in post directory for making
 # with 'make all': 
present_post_files = $(notdir $(basename $(wildcard $(sourcedir)/post/*.f90)))

#---------------------------------------------------------------------------------
 #Rules:
energy: $(objects) $(fft_lib) $(sourcedir)/post/energy.f90
	$(f90) $(fft_lib) parameters.o constants.o spectral.o $(sourcedir)/post/energy.f90 -o energy $(flags)

vspectrum: $(objects) ~/hydra/lib/stafft/stafft.f90 $(sourcedir)/post/vspectrum.f90
	$(f90) ~/hydra/lib/stafft/stafft.f90 parameters.o constants.o $(sourcedir)/post/vspectrum.f90 -o vspectrum $(flags)

accel: $(objects) $(fft_lib) $(sourcedir)/post/accel.f90
	$(f90) $(fft_lib) parameters.o constants.o spectral.o $(sourcedir)/post/accel.f90 -o accel $(flags)

variability: $(objects) $(fft_lib) $(sourcedir)/post/variability.f90
	$(f90) $(fft_lib) parameters.o constants.o spectral.o $(sourcedir)/post/variability.f90 -o variability $(flags)

gamma-tilde: $(objects) $(fft_lib) $(sourcedir)/post/gamma-tilde.f90
	$(f90) $(fft_lib) parameters.o constants.o spectral.o $(sourcedir)/post/gamma-tilde.f90 -o gamma-tilde $(flags)

hspectrum: $(objects) $(fft_lib) $(sourcedir)/post/hspectrum.f90
	$(f90) $(fft_lib) parameters.o constants.o spectral.o $(sourcedir)/post/hspectrum.f90 -o hspectrum $(flags)

slice: $(objects) $(sourcedir)/post/slice.f90
	$(f90) parameters.o constants.o $(sourcedir)/post/slice.f90 -o slice $(flags)

g2c: $(objects) $(sourcedir)/post/g2c.f90
	$(f90) parameters.o constants.o $(sourcedir)/post/g2c.f90 -o g2c $(flags)

vertical-velocity: $(objects) $(sourcedir)/post/vertical-velocity.f90
	$(f90) parameters.o constants.o $(sourcedir)/post/vertical-velocity.f90 -o vertical-velocity $(flags)

nh-pressure: $(objects) $(sourcedir)/post/nh-pressure.f90
	$(f90) parameters.o constants.o $(sourcedir)/post/nh-pressure.f90 -o nh-pressure $(flags)

 # Phony definitions:
.PHONY: post_all
 # Rule for 'make all' in the main make file:
post_all: $(present_post_files)

