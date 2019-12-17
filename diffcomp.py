#!/usr/bin/env python

# Code that reads two ascii datasets having the same structure 
# then calculates the maximum difference over all rows (times)
# for a given column (default: 2).

#=====perform the various imports========
import sys,os,warnings
import numpy as np
import argparse
warnings.simplefilter("ignore",DeprecationWarning)

#========================================
#=====various function definitions=======
#========================================

def parse_args():
  parser = argparse.ArgumentParser(prog='diffcomp')
  parser.add_argument('input', metavar='input_filename' , type=str, nargs=2,  help='Input files (exactly 2)')
  parser.add_argument('-column', metavar='column' , type=int, default = 2, help='Column to compare (default 2)')
  args = parser.parse_args() 
  return args

def get_data(fname,args):
  # Open input file(s):
  try:
     in_file = open(fname,'r')# try opening passed filename  
  except IOError, message:# error if file not found 
     print >> sys.stderr, 'File could not be opened', message
     sys.exit()

  #  Read in data and close input file:
  raw_data = np.loadtxt(in_file,dtype=float,unpack=True)
  in_file.close()

  ncol = raw_data.shape[0]
  nrow = raw_data.shape[1]
  shaped_data = raw_data.reshape((ncol,nrow))

  # Column:
  col = args.column-1

  return np.array(shaped_data[col])

def main():
  args = parse_args()

  # Read first data file:
  fname=args.input[0]
  main_array1 = get_data(fname,args)

  # Read second data file:
  fname=args.input[1]
  main_array2 = get_data(fname,args)

  diffmax=np.amax(abs(main_array2-main_array1))

  print ' Maximum absolute difference = ',diffmax

if __name__ == '__main__':
  main()
