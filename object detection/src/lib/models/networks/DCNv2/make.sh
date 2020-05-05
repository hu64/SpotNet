export PATH=/store/dev/cuda-9.0/bin${PATH:+:${PATH}}
export CPATH=/store/dev/cuda-9.0/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/store/dev/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#!/usr/bin/env bash
cd src/cuda

# compile dcn
nvcc -c -o dcn_v2_im2col_cuda.cu.o dcn_v2_im2col_cuda.cu -x cu -Xcompiler -fPIC
nvcc -c -o dcn_v2_im2col_cuda_double.cu.o dcn_v2_im2col_cuda_double.cu -x cu -Xcompiler -fPIC

# compile dcn-roi-pooling
nvcc -c -o dcn_v2_psroi_pooling_cuda.cu.o dcn_v2_psroi_pooling_cuda.cu -x cu -Xcompiler -fPIC
nvcc -c -o dcn_v2_psroi_pooling_cuda_double.cu.o dcn_v2_psroi_pooling_cuda_double.cu -x cu -Xcompiler -fPIC

cd -
python build.py
python build_double.py
