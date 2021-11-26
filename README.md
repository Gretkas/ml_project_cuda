# ml_project_cuda
An implementation of Ojas learning rule written in cuda. The whole implementation will run on the GPU. 
Read the corresponding report for futher information

## Requirements
- Nvidia GPU supporting CUDA
- Proprietairy Nvidia driver
- CUDA toolkit (nvcc compiler)

## Setup and running
1. git clone
2. cd to `ml_project_cuda`
3. compile with nvcc using this command:
```bash
nvcc main.cu -rdc=true -o ojas
``` 
3. run the code with profiler using this command: 

```bash
nvprof ./ojas --profile-from-start off
``` 

