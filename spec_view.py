#!/usr/bin/env python

# This script plots spectra for vorticity, divergence and acceleration
# divergence from data in spectra.asc.

# Further options can be found using spec_view -h

#=====perform the various imports========
#=======needed by the main code==========
import sys,os,warnings
# Define main hydra tree locations:
uname=os.getlogin()
homedir=os.getenv('HOME')
rootdir=os.path.join(homedir,'hydra','scripts')
moddir=os.path.join(rootdir,'modules')
graphicsdir=os.path.join(rootdir,'graphics')
sys.path.append(moddir)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
# Set default plot save resolution to a large value:
mpl.rcParams['savefig.dpi'] = 200
# Set label size:
mpl.rcParams['xtick.labelsize'] = 30
mpl.rcParams['ytick.labelsize'] = 30
mpl.rcParams['axes.linewidth'] = 3
import argparse
warnings.simplefilter("ignore",DeprecationWarning)

#========================================
#=====various function definitions=======
#========================================

def parse_args():
  # Define a parser and read in the command line arguments passed when the script is run.
  parser = argparse.ArgumentParser(prog='spec-view')
  # Argument list:
  # Default limits for the spectra are -12, 0.
  # this can be over-ridden with this option:
  parser.add_argument('-xlims', metavar='x_lim' , type=float , nargs=2, default='0.0 2.25'.split(),help='Lower and upper limits for x range of the plot')
  parser.add_argument('-ylims', metavar='y_lim' , type=float , nargs=2, default='-12.0 0.0'.split(),help='Lower and upper limits for y range of the plot')
  args = parser.parse_args() 
  # Return a parsed argument object:
  return args

def running(args):
  # Main code opening and controlling a plotting window.

  # Open input file:
  in_file=open('spectra.asc','r')
  # Read the first header line to get kmax:
  first_line=in_file.readline()
  kmax=int(first_line.split()[-1])
  in_file.seek(0)

  nx=kmax

  # Read in the full data to a 1d array and close input file:
  raw_data = np.fromfile(file=in_file,dtype=float,sep='\n')
  in_file.close()

  # Set the number of frames:
  nframes = int(len(raw_data)/(4*kmax+2))  
  print 'Number of frames found %i' %nframes

  # Shape the data array into a useful shape for plotting:
  frames=range(0,nframes)
  time=[raw_data[i*(4*kmax+2)] for i in frames]
  tim_eles = [i*(4*kmax+2)+j for i in frames for j in range(2)]
  shaped_data = np.delete(raw_data,tim_eles)[0:(4*kmax+2)*nframes].reshape((nframes,kmax,4))
  k=np.zeros((nframes,nx))
  z=np.zeros((nframes,nx))
  d=np.zeros((nframes,nx))
  g=np.zeros((nframes,nx))
  for i in frames:
    k[i,:]=shaped_data[i].transpose()[0][0:nx]
    z[i,:]=shaped_data[i].transpose()[1][0:nx]
    d[i,:]=shaped_data[i].transpose()[2][0:nx]
    g[i,:]=shaped_data[i].transpose()[3][0:nx]
  global ic 
  # Grab the correct sub-array for plotting (ic is the current frame)  
  ic = 0
  # Initiate a plotting window and plot relevant data into it:
  fig = plt.figure(1,figsize=[12.5,12])
  fig.subplots_adjust(left=0.15)
  ax = fig.add_subplot(111)
  im  = ax.plot(k[ic],z[ic],'b-',lw=3,label='$\\zeta$')
  im2 = ax.plot(k[ic],d[ic],'r-',lw=3,label='$\\delta$')
  im3 = ax.plot(k[ic],g[ic],'m-',lw=3,label='$\\gamma$')
  # Set plot title, legend and limits of the plot:
  ax.set_title('Spectra at $t =$'+str('%.2f'%time[ic]), fontsize=40)
  ax.set_xlabel('$\log_{10}\,k$', fontsize=40)
  ax.set_ylabel('$\log_{10}\,|\hat{a}|^2$', fontsize=40)
  ax.tick_params(length=10, width=3)
  ax.legend(loc='lower left',prop={'size':30}, shadow=True)
  xlimits=[float(x) for x in args.xlims]
  ax.set_xlim(xlimits)
  ylimits=[float(y) for y in args.ylims]
  ax.set_ylim(ylimits)

  def on_press(event):
    # Routine to deal with re-plotting the window on keypress. 
    # The keys -/= (ie. -/+ without the shift key) cycle through frames 
    # forward and backwards.
    global ic
    if event.key=='=':
      axes=event.canvas.figure.get_axes()[0]
      ic = (ic+1+nframes) % nframes
      xlim=axes.get_xlim()
      ylim=axes.get_ylim()
      # Clear the axes for replot:
      axes.clear()
      # Replot the relevant data:
      axes.plot(k[ic],z[ic],'b-',lw=3,label='$\\zeta$')
      axes.plot(k[ic],d[ic],'r-',lw=3,label='$\\delta$')
      axes.plot(k[ic],g[ic],'m-',lw=3,label='$\\gamma$')
      # Set the title, legend and plot limits:
      axes.set_title('Spectra at $t =$'+str('%.2f'%time[ic]), fontsize=40)
      axes.set_xlabel('$\log_{10}\,k$', fontsize=40)
      axes.set_ylabel('$\log_{10}\,|\hat{a}|^2$', fontsize=40)
      axes.tick_params(length=10, width=3)
      axes.legend(loc='lower left',prop={'size':30}, shadow=True)
      axes.set_xlim(xlim)  
      axes.set_ylim(ylim)
      plt.draw()
    if event.key=='-':
      axes=event.canvas.figure.get_axes()[0]
      ic = (ic-1+nframes) % nframes
      xlim=axes.get_xlim()
      ylim=axes.get_ylim()
      # Clear the axes for replot:
      axes.clear()
      # Replot the relevant data:
      axes.plot(k[ic],z[ic],'b-',lw=3,label='$\\zeta$')
      axes.plot(k[ic],d[ic],'r-',lw=3,label='$\\delta$')
      axes.plot(k[ic],g[ic],'m-',lw=3,label='$\\gamma$')
      # Set the title, legend and plot limits:
      axes.set_title('Spectra at $t =$'+str('%.2f'%time[ic]), fontsize=40)
      axes.set_xlabel('$\log_{10}\,k$', fontsize=40)
      axes.set_ylabel('$\log_{10}\,|\hat{a}|^2$', fontsize=40)
      axes.tick_params(length=10, width=3)
      axes.legend(loc='lower left',prop={'size':30}, shadow=True)
      axes.set_xlim(xlim)  
      axes.set_ylim(ylim)
      # Finally redraw the plot:
      plt.draw()

  # Begin the plot and link keypress events to the above handler:
  cid=fig.canvas.mpl_connect('key_press_event',on_press)
  plt.show()
 

if __name__ == '__main__':
  # Main code to drive the above routines:
  args = parse_args()
  running(args)
