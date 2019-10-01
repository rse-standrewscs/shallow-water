 # Set existence of directory variable used in main makefile:
init_exists = true
 # Calculate f90 codes existing in init directory for making
 # with 'make all': 
present_init_files = $(notdir $(basename $(wildcard $(sourcedir)/init/*.f90)))

#---------------------------------------------------------------------------------
 #Rules:
vstrip: $(objects) $(sourcedir)/init/vstrip.f90
	$(f90) parameters.o constants.o $(sourcedir)/init/vstrip.f90 -o vstrip $(flags)

swto3d: $(objects) $(sourcedir)/init/swto3d.f90
	$(f90) parameters.o constants.o $(sourcedir)/init/swto3d.f90 -o swto3d $(flags)

threevort: $(objects) $(sourcedir)/init/threevort.f90
	$(f90) parameters.o constants.o $(sourcedir)/init/threevort.f90 -o threevort $(flags)

balinit: $(objects) $(fft_lib) $(sourcedir)/init/balinit.f90
	$(f90) $(fft_lib) parameters.o constants.o spectral.o $(sourcedir)/init/balinit.f90 -o balinit $(flags)

 # Phony definitions:
.PHONY: init_all
 # Rule for 'make all' in the main make file:
init_all: $(present_init_files)


