# Interpolator-network
A simple resolution independent  frame interpolator  
# Background info
This one is an indirect network implementation of 3d Voxel flow paper (google it!). I removed the trilinear sampler and added SSIM, L1, L2 losses and the interpolation works fine. I will upload more results later.
Additional stuff : The networks.py also stores some of the used network functions. Any addition is welcome. 

# Rough training information

The dataset has been put into a single folder. It has all the frames within it and the only way to identify different frames is to see the end number of the file (*_00.ppm, *_01.ppm, *_02.ppm etc.). The end numbers provide the frame index.

Assuming for current scenario, we are taking the all_frames folder in graphics/scratch keeping it intact.

1. Keeping the aforesaid assumptions in mind, we first convert all our corresponding frames into a lmdb file for smooth dataflow. In the program 'dataflow.py', the name of the folder is to be selected, whether it should be 1k or 2k.

2. If it is 2k, select the HEIGHT and WIDTH parameter to 512, for 1k it should be 256.

3. Note the folder space where the lmdb file is stored. It should be at the end of the program.

4. We move on to the 'Interpolator.py' module. The address of the training data (lmdb) module should be noted and put into place so that it can train properly. The training files or the checkpoints will be saved automatically in a folder called 'train_log' once the training starts.

5. A sub-module called the 'networks.py' stores the networks used in this case. Use the network 'Voxeldeep' in 'Interpolator.py' for 1k image and 'Voxeldeeper' for 2k images. It should be somewhere around Line number 68 in 'Interpolator.py' file as a function name.

6. After sufficient traing has been achieved, proceed with the prediction algorithm. In the file 'predict_1k_2k.py' one should be able to change the options and run the file for both the 1k and 2k images. The argument section in the main() section of the program provides the necessary backdrop required. While running 2k, it is adviced not to provide Hight and Width parameter as it is already set to maximum limit the current gpu configuration can handle.

7. Additionally, one can also view the ongoing results in tensorboard, while running the bash command inside the folder './training_log/interpolator/':

python -m tensorboard.main --logdir=.  

Note: Bash command might vary with tensorflow configuration and version (current one is tf 1.5)
