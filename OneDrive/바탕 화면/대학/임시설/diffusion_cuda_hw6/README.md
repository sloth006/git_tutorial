# Emb26Spring: Diffusion Inference on Embedded Devices

As many embedded devices have limited resources, they even fail to run PyTorch C++ (torchlib). Therefore, we sometimes need to run ML models with a native C++ implementation. This project implements the diffusion inference with pure C++ without the PyTorch. This project runs basic DDPM and fast sampling with DDIM.

This project utilizes the following 3rd party libraries.
- argparse (https://github.com/p-ranav/argparse): argument parser.
- tqdm c++17 (https://github.com/mraggi/tqdm-cpp): for progress bar display.

## Build
Just run `sh scripts/build.sh` (do not make a slurm script for the build).It will generate the cross-compiled executable for the Jetson board. Do not execute the executable on the login server, it will not work because the executable is targeted for the arm64 architecture.

## For Automatic Run (Sufficient for the Project)
To run the full procedures, including the test containing the unit test and the end-to-end functionality test, just run the `sbatch test.sbatch` after the build. For sampling only, please run `sbatch run.sbatch` and check the `sampled_img_0.png` in the `results` directory. You can check the running time of your implementation in the slurm result file (`logs/{run/test}_err.diff_inf.{JOBID}`). Other logs will be saved at the `logs/{run/test}_out.diff_inf.{JOBID}`.

## Where to Accelerate
The project's main operations are gathered in `src/include/functions.hpp`. All modules reuse the functions in this file with function pointers. Therefore, as a first step, please accelerate the functions in this file. We denoted the main acceleration targets with the comment `[TODO] — accelerate`. However, you can modify other functions in `functions.hpp`, the `AttentionUNet` class and `modules.h/cpp` for other accelerations (e.g., operation fusion). But, make sure to fulfill the end-to-end functionality bar. We check this by checking whether the cosine similarity is larger than 0.995 (you can check this by running `sbatch test.sbatch`). Also, do not modify the sampling procedure (i.e., the `DiffusionModel` class and `schedules.hpp`).

## Running Time of Naive Implementation
The naive implementation with the default setting (64x64 and DDIM 10x) takes around 211-220 seconds on the NVIDIA Jetson Orin Nano.


## Other Detailed Explanations
### Sampling
For the detailed arguments, please run `bin/sample -h` by changing the scripts. We omit the explanation of the arguments because they are already explained in the helper (`-h`).

### Image Generation
After running the `bin/sample`, `initial_images_bchw.npy` and `sampled_images_bchw.npy` will be saved. After move to the utils directory (`cd utils`), run `/nfs/home/proj1_env/bin/python3 img_convert.py`. This procedure will generate two `.png` images: the initial image (random noise, `initial_img_{i}.png`) and the generated image (`sampled_img_{i}.png`). For doing the project, you do not need to care about it because we already made scripts for this procedure. For already generated example images, please check the `examples` directory.

### Unit Functionality Check
For the functionality check (unit test), run `bin/functionality` by changing the scripts. It will show the test results of your code's functionality. However, it does not guarantee the full functionality of your code because it is just a helper. Also, we already made scripts for this check.

## License
Do not distribute or share this code. It is only for the projects of the embedded systems designs lecture in SNU. Jaeyong Song wrote the base code structure.
