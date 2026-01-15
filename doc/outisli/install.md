鉴于大家可能觉得从源码安装`DeePMD-kit`门槛较高，而极少使用。然而从源码安装的灵活性最高，为进一步推广，并减少可能的坑，笔者在此根据自己的安装流程结合官方文档给出一个适用性较广的安装教程，各位可自行尝试。

本教程适用于 Linux(with NVIDIA GPU) 及 Mac(with Apple Silicon)

Since some users may find installing `DeePMD-kit` from source to be challenging and rarely attempt it, this guide aims to make the process more accessible. Installing from source offers the highest flexibility. To promote this method and reduce potential pitfalls, I have compiled a broadly applicable installation tutorial based on my own experience and the official documentation. You are encouraged to try it out.

This tutorial is applicable to Linux (with NVIDIA GPU) and Mac (with Apple Silicon).

> 注：
>
> 1. 安装过程不强制要求`sudo`权限
> 1. 若在有`sudo`权限的电脑上，可自行安装 CUDA Toolkit 以及 mpi（可选）
> 1. 在 HPC 集群上可通过`source\module`方式加载 CUDA Toolkit 以及 mpi 环境
> 1. 默认安装在用户 home 目录 Software 目录下，若需要修改路径，请修改教程中涉及路径的命令
> 1. 本教程需有一定计算机（linux）操作常识，若遇到问题，可以评论沟通或询问 AI

> Notes:
>
> 1. The installation process does not strictly require `sudo` privileges.
> 1. If you have `sudo` privileges, you may install CUDA Toolkit and MPI (optional) yourself.
> 1. On HPC clusters, you can load CUDA Toolkit and MPI environments using `source` or `module` commands.
> 1. By default, the installation path is set to the user's home directory under the Software folder. If you wish to change the path, please modify the relevant commands in the tutorial.
> 1. This tutorial assumes some basic knowledge of computer (Linux) operations. If you encounter any issues, feel free to comment or ask AI for help.

# 0. Preparation (Optional)

## 0.1 CUDA Toolkit

```shell
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y
# for Ubuntu 24.04 LTS
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y
sudo apt install cuda-toolkit-12-8 -y

#config cuda
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/extras/CUPTI/lib64
```

You can also use CUDA 12.6, 12.9 as well.

## 0.2 Intel® oneAPI Toolkit

```shell
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/e6ff8e9c-ee28-47fb-abd7-5c524c983e1c/l_BaseKit_p_2024.2.1.100_offline.sh
sudo sh ./l_BaseKit_p_2024.2.1.100_offline.sh -a --silent --cli --eula accept

wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/d461a695-6481-426f-a22f-b5644cd1fa8b/l_HPCKit_p_2024.2.1.79_offline.sh
sudo sh ./l_HPCKit_p_2024.2.1.79_offline.sh -a --silent --cli --eula accept

# load intel oneapi
source /opt/intel/oneapi/setvars.sh --force > /dev/null
```

# 1. Install Backend’s Python interface

## 1.1 Use Miniforge (Conda/mamba)

```shell
# 0. no need for HPC
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y

# 1. Preparation
# 1.1 Get source code
# or git clone https://github.com/deepmodeling/deepmd-kit.git && cd deepmd-kit && git checkout devel
# 下述链接为笔者自己的fork，时不时增加一些小改进，欢迎star
git clone git@github.com:OutisLi/deepmd-kit.git && cd deepmd-kit && git checkout outisli

# 1.2 Create virtual environment
# optional if you installed miniforge: alias mamba="conda"
# CUDA 13.0 support gcc-15
mamba update -n base -c conda-forge conda -y ; mamba update -n base -c conda-forge mamba -y
mamba deactivate && mamba env remove -n dpmd -y ; rm -rf build ; git clean -xdf ; mamba create -n dpmd gcc=15 gxx=15 cmake python=3.13 -c conda-forge -y && mamba activate dpmd && pip install --upgrade pip && pip install uv

# 1.3 Open MPI with UCX: a CUDA-aware MPI for LAMMPS on several GPUs, whose CUDA support a run on several GPUs enables at launch (section 3)
mamba install -c conda-forge openmpi ucx -y

# 2.1 Install pytorch
uv pip install -U torch --index-url https://download.pytorch.org/whl/cu130

# 2.2 (Optional) Install tensorflow
uv pip install -U tensorflow

# 2.3 (Optional) Install jax
uv pip install -U "tensorflow[and-cuda]" "jax[cuda13]" jax-ai-stack equinox

# 3. Install deepmd-kit
export CUDA_VERSION=13.1 CUDA_HOME="/usr/local/cuda" && export CUDAToolkit_ROOT=$CUDA_HOME CUDA_PATH=$CUDA_HOME && export DP_VARIANT="cuda" DP_ENABLE_PYTORCH=1 DP_ENABLE_TENSORFLOW=1 DP_ENABLE_PADDLE=0 DP_ENABLE_NATIVE_OPTIMIZATION=1 && pip install -e . -v
# 3.1 (Optional, developers) the float64 DPA4 training kernels serve only the float64 consistency tests and are off by default.
# Compile them with the line below; the build tree caches the choice, so pass -DDEEPMD_ENABLE_DPA4_FP64=OFF once to return to the default.
# export CMAKE_ARGS="-DDEEPMD_ENABLE_DPA4_FP64=ON" && pip install -e . -v

# 4.1 Install other useful packages
uv pip install -U dpdata pymatgen freud-analysis seaborn ipykernel nglview "git+https://gitlab.com/1041176461/ase-abacus.git"
# 4.2 For developers
uv pip install -U pytest pre-commit tensorboard torch-tb-profiler tensorboard-plugin-profile
```

### 1.1+ Check GPU Installation

```shell
# pytorch
python -c "import torch; print('PyTorch devices:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else 'CPU')"

# tensorflow
python -c "import tensorflow as tf; print('TF devices:', tf.config.list_physical_devices('GPU'))"

# JAX
python -c "import jax; print('JAX devices:', jax.devices())"

# All in One
python -c "import torch, tensorflow as tf, jax; print('PyTorch: ', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else 'CPU'); print('TF:      ', tf.config.list_physical_devices('GPU')); print('JAX:     ', jax.devices())"
```

## 1.2 For Mac

```shell
# 1. Preparation
# 1.1 Get source code
# or git clone https://github.com/deepmodeling/deepmd-kit.git
git clone git@github.com:OutisLi/deepmd-kit.git && cd deepmd-kit && git checkout outisli
# 1.2 Create virtual environment
conda update -n base -c conda-forge conda -y ; conda update -n base -c conda-forge mamba -y
conda deactivate && conda env remove -n dpmd -y ; rm -rf build ; git clean -xdf ; mamba create -n dpmd compilers llvm-openmp python=3.13 -c conda-forge -y && mamba activate dpmd && pip install --upgrade pip && pip install uv

# 2. Install pytorch
uv pip install -U torch

# 3. Install deepmd-kit
export DP_ENABLE_PYTORCH=1 DP_ENABLE_PADDLE=0 DP_ENABLE_TENSORFLOW=0 DP_ENABLE_NATIVE_OPTIMIZATION=1 && uv pip install -e . -v

# 4.1 Install other useful packages
uv pip install -U dpdata pymatgen freud-analysis seaborn ipykernel nglview "git+https://gitlab.com/1041176461/ase-abacus.git"
# 4.2 For developers
uv pip install -U pytest pre-commit tensorboard torch-tb-profiler tensorboard-plugin-profile
```

# 2. Install the C++ interface

> If one does not need to use DeePMD-kit with LAMMPS or i-PI, then the python interface installed in the previous section does everything and he/she can safely skip this section.

```shell
# 0. (Optional) for reinstall
export software="$HOME/Software" && rm -rfv $software/deepmd-kit_cpp $software/deepmd-kit/source/build

# 1. Environment Variables
export deepmd_source_dir=$(pwd) && mkdir -p ../deepmd-kit_cpp && cd ../deepmd-kit_cpp && export deepmd_root=$(pwd) && cd ../deepmd-kit && cd source && mkdir -p build && cd build
# export deepmd_source_dir="$software/deepmd-kit"
# export deepmd_root="$software/deepmd-kit_cpp"

# 2. CMake (Choice either one)

# 2.1 Option 1: use pytorch & tensorflow & jax (from python env)
cmake -DCMAKE_INSTALL_PREFIX=$deepmd_root \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_NATIVE_OPTIMIZATION=ON \
      -DUSE_CUDA_TOOLKIT=ON \
      -DENABLE_PYTORCH=ON \
      -DUSE_PT_PYTHON_LIBS=ON \
      -DENABLE_TENSORFLOW=ON \
      -DUSE_TF_PYTHON_LIBS=ON \
      -DENABLE_JAX=ON \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..

# 2.2 Option 2: use pytorch only (from python env)
cmake -DCMAKE_INSTALL_PREFIX=$deepmd_root \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_NATIVE_OPTIMIZATION=ON \
      -DUSE_CUDA_TOOLKIT=ON \
      -DENABLE_PYTORCH=ON \
      -DUSE_PT_PYTHON_LIBS=ON \
      -DENABLE_TENSORFLOW=OFF \
      -DUSE_TF_PYTHON_LIBS=OFF \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..

# 2.3 Option 3: use libtorch (standalone)
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
# Note: $software/libtorch is the unzipped dir, CMAKE_INSTALL_PREFIX is set to a local dir
cmake -DCMAKE_INSTALL_PREFIX="../../install" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_NATIVE_OPTIMIZATION=ON \
      -DUSE_CUDA_TOOLKIT=ON \
      -DENABLE_PYTORCH=ON \
      -DUSE_PT_PYTHON_LIBS=OFF \
      -DCMAKE_PREFIX_PATH=$software/libtorch ..

# 3. Install
make -j && make install

# 4. (Optional) Link cmake cache
rm $software/deepmd-kit/compile_commands.json ; ln -s "$(pwd)/compile_commands.json" $software/deepmd-kit
```

`make install` and `cmake --install` write every installed file in place. A process that runs LAMMPS or the C++ interface keeps the libraries of `$deepmd_root` mapped, and rewriting a mapped library under it can crash it, so while such jobs run, stage the install and move the files into place instead: each `mv` replaces a file by a rename, which leaves the running processes the old copy. The staging directory must lie on the same filesystem as `$deepmd_root`, so that every `mv` is a rename. The two symbolic links `lib/libdeepmd_cc_low.so` and `lib/deepmd_lmp/dpplugin.so` bypass the stage: the install steps that make them write to `$deepmd_root` itself, even under `DESTDIR`, and re-create them there with unchanged targets.

```shell
DESTDIR=$software/.deepmd-kit_cpp_stage cmake --install .
(cd $software/.deepmd-kit_cpp_stage$deepmd_root && find . -type d -exec mkdir -p "$deepmd_root/{}" \; && find . \( -type f -o -type l \) -exec mv -f "{}" "$deepmd_root/{}" \;)
rm -rf $software/.deepmd-kit_cpp_stage
```

# 3. Install LAMMPS with the DeePMD-kit styles

_Before following this section, the [DeePMD-kit C++ interface](https://docs.deepmodeling.com/projects/deepmd/en/master/install/install-from-source.html) of section 2 should have been installed._

LAMMPS loads the DeePMD-kit pair styles at startup from `libdeepmd_lmp.so`, which the C++ interface builds and installs once it is configured against that LAMMPS. LAMMPS itself therefore carries no DeePMD-kit code, and a change of the DeePMD-kit sources, the pair styles included, needs only the C++ interface rebuilt. The plugin is compiled against the headers and the Kokkos configuration of one LAMMPS build, so it is rebuilt whenever that LAMMPS is rebuilt or upgraded.

The Kokkos package of a LAMMPS build tree looks for its compiler launcher and `nvcc_wrapper` in the install prefix, where LAMMPS does not install them, so step 2 names both.

Note on GPU architecture: `Kokkos_ARCH_*` must match the card.

> Common values: Pascal (GTX 1080, Titan X): `Kokkos_ARCH_PASCAL61` Volta (V100): `Kokkos_ARCH_VOLTA70` Turing (RTX 20xx, T4): `Kokkos_ARCH_TURING75` Ampere: `Kokkos_ARCH_AMPERE80` (A100) or `Kokkos_ARCH_AMPERE86` (RTX 30xx) Lovelace (RTX 40xx): `Kokkos_ARCH_ADA89` Hopper (H100, H20): `Kokkos_ARCH_HOPPER90` Blackwell: `Kokkos_ARCH_BLACKWELL100` (B200) or `Kokkos_ARCH_BLACKWELL120` (RTX 50xx, RTX PRO 6000)

```shell
# 0. Environment: Kokkos compiles with nvcc_wrapper, which calls the nvcc on PATH
export software="$HOME/Software" && export deepmd_source_dir="$software/deepmd-kit" && export deepmd_root="$software/deepmd-kit_cpp"
export PATH="$CONDA_PREFIX/bin:/usr/local/cuda/bin:$PATH" && export NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++

# 1. Download LAMMPS
cd $software && mkdir -p lammps && cd lammps && export version="patch_4Jul2026" && wget "https://gh-proxy.com/github.com/lammps/lammps/archive/${version}.tar.gz" && tar xzf "${version}.tar.gz"
export LAMMPS_ROOT="$software/lammps/lammps-${version}"

# 2. Build LAMMPS, installed into its own source tree
# !!! CHANGE Kokkos_ARCH_BLACKWELL120 TO MATCH YOUR GPU !!!
cd $LAMMPS_ROOT
cmake -S cmake -B build_kk \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=$LAMMPS_ROOT \
      -D BUILD_SHARED_LIBS=yes \
      -D LAMMPS_INSTALL_RPATH=ON \
      -D BUILD_MPI=yes \
      -D BUILD_OMP=yes \
      -D PKG_KOKKOS=ON \
      -D Kokkos_ENABLE_SERIAL=ON \
      -D Kokkos_ENABLE_CUDA=ON \
      -D Kokkos_ARCH_BLACKWELL120=ON \
      -D CMAKE_CXX_COMPILER=$LAMMPS_ROOT/lib/kokkos/bin/nvcc_wrapper \
      -D PKG_PLUGIN=ON \
      -D PKG_KSPACE=ON \
      -D PKG_MANYBODY=ON \
      -D PKG_MEAM=ON \
      -D PKG_SPIN=ON \
      -D FFT_KOKKOS=KISS
cmake --build build_kk -j 64 && cmake --install build_kk

# 3. The plugin: configure the C++ interface of section 2 against that LAMMPS, then rebuild and install it
cd $deepmd_source_dir/source/build
cmake -D LAMMPS_SOURCE_ROOT=$LAMMPS_ROOT \
      -D DEEPMD_LAMMPS_KOKKOS=ON \
      -D Kokkos_DIR=$LAMMPS_ROOT/build_kk/lib/kokkos \
      -D Kokkos_COMPILE_LAUNCHER=$LAMMPS_ROOT/lib/kokkos/bin/kokkos_launch_compiler \
      -D Kokkos_NVCC_WRAPPER=$LAMMPS_ROOT/lib/kokkos/bin/nvcc_wrapper ..
cmake --build . -j 64 && cmake --install .

# 4. Run: LAMMPS loads every *plugin.so of LAMMPS_PLUGIN_PATH at startup
# (or put `plugin load $deepmd_root/lib/libdeepmd_lmp.so` at the top of the input),
# and the RUNPATH of the plugin finds the libraries of $deepmd_root and of torch
export LAMMPS_PLUGIN_PATH=$deepmd_root/lib/deepmd_lmp
$LAMMPS_ROOT/bin/lmp -k on g 1 -sf kk -in in.lammps
# Several GPUs: the CUDA support of Open MPI is enabled for that launch only,
# since on a single rank its memory-type query of every MPI call slows small systems
mpirun --mca opal_cuda_support 1 -x UCX_MEMTYPE_CACHE=n -np 2 $LAMMPS_ROOT/bin/lmp -k on g 2 -sf kk -in in.lammps
```

The styles can also be compiled into the public release instead of the plugin: append `include($deepmd_source_dir/source/lmp/builtin.cmake)` to `cmake/CMakeLists.txt` before step 2 of section 3, add `-DCMAKE_PREFIX_PATH="$deepmd_root;$CONDA_PREFIX;$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"`, and skip its step 3; such a LAMMPS runs without `LAMMPS_PLUGIN_PATH` as well. Both ways link the same installed DeePMD-kit libraries and run at the same speed: on an RTX PRO 6000 Blackwell every DPA4C grade, plain and ZBL-bridged, steps within 0.5% of the other way from 250 thousand atoms up.

## 3.1 The OutisLi fork with native spin

The Newtonian spin integrators `fix nve/spin`, `fix nve/spin/kk` and the Nose-Hoover extension of the OutisLi fork of LAMMPS (`NEWTONIAN_SPIN.md` there) integrate the raw magnetic force `-dE/dS`, which `pair_style dpa4spin` and `dpa4spin/kk` of the DeePMD-kit plugin write under `magnetic_force cartesian`. The fork builds as the public release of section 3 and loads a plugin of its own, because a plugin is compiled against the headers and the Kokkos package of one LAMMPS tree. That plugin comes from a plugin-only build tree inside the fork, which links the C++ interface already installed in `$deepmd_root` (`DEEPMD_C_ROOT`) and is installed into the fork's `lib`.

```shell
# 1. Get the fork and build it as step 2 of section 3
cd $software/lammps && git clone -b outisli https://github.com/OutisLi/lammps.git lammps-outisli
export LAMMPS_ROOT="$software/lammps/lammps-outisli"

# 2. The plugin of the fork, against the C++ interface of $deepmd_root and the MPI of the LAMMPS build
cmake -S $deepmd_source_dir/source -B $LAMMPS_ROOT/build_deepmd \
      -D DEEPMD_C_ROOT=$deepmd_root \
      -D CMAKE_INSTALL_PREFIX=$LAMMPS_ROOT \
      -D LAMMPS_SOURCE_ROOT=$LAMMPS_ROOT \
      -D DEEPMD_LAMMPS_KOKKOS=ON \
      -D Kokkos_DIR=$LAMMPS_ROOT/build_kk/lib/kokkos \
      -D Kokkos_COMPILE_LAUNCHER=$LAMMPS_ROOT/lib/kokkos/bin/kokkos_launch_compiler \
      -D Kokkos_NVCC_WRAPPER=$LAMMPS_ROOT/lib/kokkos/bin/nvcc_wrapper \
      -D ENABLE_PYTORCH=ON -D USE_PT_PYTHON_LIBS=ON -D CMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -D MPI_C_COMPILER=$CONDA_PREFIX/bin/mpicc -D MPI_CXX_COMPILER=$CONDA_PREFIX/bin/mpicxx
cmake --build $LAMMPS_ROOT/build_deepmd -j 64 --target deepmd_lmp && cmake --install $LAMMPS_ROOT/build_deepmd/lmp/plugin

# 3. Run with magnetic_force cartesian in the input
export LAMMPS_PLUGIN_PATH=$LAMMPS_ROOT/lib/deepmd_lmp
$LAMMPS_ROOT/bin/lmp -k on g 1 -sf kk -in in.lammps
```

# 4. DPGEN2

```shell
# alias conda="mamba"
export software="$HOME/Software"
cd $software
git clone git@github.com:OutisLi/dpgen2.git
cd dpgen2 && conda activate dpmd && pip install uv dpdispatcher && uv pip install -e . -v
```
