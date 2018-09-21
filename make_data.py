from src.datagen import *

dz = 300
ft = '/work/03782/rsen/maverick2/mimicdata/dim' + str(dz) + '_random/datafile'
parallel_random_sample_gen(dx=1,dy=1,dz=dz,filetype=ft,freq=1.0,nstd=0.05,num_data=200,\
                           fixed_function=None,num_proc=32,nsamples=5000)
