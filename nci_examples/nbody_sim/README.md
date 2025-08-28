# Building 

Make sure you have cuda loaded i.e. the env variable `$CUDA_HOME` is defined, otehrwise it will die. 

Then simply: `make` 

# Running 

To run the validation, simply: `./validate` 

To run a production run, simply: `./main N_PARTICLES N_STEPS` for example a simulation of 4096 particles with a time evolution of 1000 steps would be: `./main 4096 1000`

# Profiling 

To run the nsys profiler to get a general sense of the run, do: `nsys profile --stats=true ./main N_PARTICLES N_STEPS` this has minimal overhead. 

To run the nsight compute profiler you can do :  `ncu -k "nbody_kernel" -c COUNT -s SKIP_COUNT ./main N_PARTICLES N_STEPS` where COUNT is the number of 
kernels to profile with detail, i.e. the number of times this `==PROF== Profiling "nbody_kernel": 0%....50%....100% - 19 passes` wil appear and be done 
this has a high overhead. 

Then `SKIP_COUNT` is the number of kernel launches to skip and _NOT_ profile, i.e. let the first `SKIP_COUNT` kernels run normally and then profile 
the next `COUNT` ones you'd like to do. 

Since the kernels are repetitive it is advised to only porifle one and ksip however many your heart desires. but don't skip more than there are kernels launched. 
