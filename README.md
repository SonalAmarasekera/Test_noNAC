To self: Make sure to install Miniconda and create a new environment to avoid dependency issues.  \
URL: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  \
To restart terminal: source ~/.bashrc  

Guide to open Jupyter Notebook on a remote server. https://web.archive.org/web/20200628012208/https://docs.anaconda.com/anaconda/user-guide/tasks/remote-jupyter-notebook

Check CUDA and CUDA Toolkit----  \
nvidia-smi  \
python -c "import torch; print('torch', torch.version, 'cuda', torch.version.cuda)"  \
which nvcc && nvcc --version --> if no toolkit--> conda install -c nvidia cuda-toolkit=<cuda_version> -y  \
If that doesn't work, use NVIDIA official site instructions  

Set environement variables after toolkit installation----  \
Check if available: ls -la /usr/local/cuda/include/ | grep bf16 \
export CUDA_HOME=/usr/local/cuda  \
export PATH=$CUDA_HOME/bin:$PATH  \
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH  \
export CPATH=$CUDA_HOME/include:$CPATH  
