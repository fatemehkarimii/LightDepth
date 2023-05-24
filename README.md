![image](https://github.com/fatemehkarimii/LightDepth/assets/49230804/df7db4d2-11d1-4454-878d-54ecb1bb6de4) 

* A simple Tensorflow/PyTorch Implementation of the "LightDepth: A Resource Efficient Depth Estimation Approach for Dealing with Ground Truth Sparsity via Curriculum Learning" paper. The paper can be read [here](https://arxiv.org/abs/2211.08608).
## Results
* Visual comparison to demonstrate the improvement of our output over DenseNet on KITTI dataset. The left column presents the input and the right column presents sparse ground truth depth maps.
![image](https://github.com/fatemehkarimii/LightDepth/assets/49230804/37e2ad99-da7d-4615-ad97-5968b4048396)
* Performance comparisons of state-of-the-art depth estimation models on the KITTI Eigen split dataset\cite{eigen}. The Raspberry Pi 4 device was used for evaluation, considering trainable parameters (Params), Gflops, Runtime, and Battery. Results are reported in millions of parameters, seconds for runtime, and Watt Seconds (WS) for battery. The table highlights the best results in bold.
![image](https://github.com/fatemehkarimii/LightDepth/assets/49230804/34eb94fc-1e6f-45e6-8b06-773f82843765)


## Data
* [KITTI](https://www.cvlibs.net/datasets/kitti/): copy the raw data to a folder with the path '../kitti'. Our method expects dense input depth maps, therefore, you need to run a depth inpainting method on the Lidar data. 
## Training & Evaluation
* Simply run python main.py.

