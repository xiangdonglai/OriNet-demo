# OriNet-demo
This is the testing code for OriNet on MPI-INF-3DHP test set. Traing code will be released later.

The model was first pretrained on MPII and Human3.6M dataset and then fine-tuned on MPI-INF-3DHP training set *without* using any background augmentation.

The pretrained model can be downloaded [here](https://drive.google.com/file/d/16UM6X5-7Drat2q1xF1BnvwslkFwsGNti/view?usp=sharing)

##### Dependencies:
* [Torch 7](http://torch.ch/docs/getting-started.html#_)
* hdf5
* cudnn

#### Testing
- Download the [MPI-INF-3DHP dataset](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) and [annotation file](https://drive.google.com/file/d/1R1hWjRDHB8IOQZ74RIBOWA3_m_l6llVS/view?usp=sharing) and put the h5 file in `data/mpi/`
- Suppose that the test sequences are located in *mpi-inf-3dhp/test/*
- In directory `src`, run `th test.lua -dataDir mpi-inf-3dhp -loadModel /path/to/your/model` to save the results in `Result.txt`
- Go to directory`test_util` and run `evaluate.m` in matlab to get the evaluation results. Remember to change the path to your annotation files. (The evaluation codes are provided in MPI-INF-3DHP dataset)

##### Results on MPI-INF-3DHP test set by activities

|      | Stand/ Walk | Exercies | Sit on  Chair | Crouch/ Reach | On the  Floor | Sports | Misc. | All PCK | All AUC | MPJPE(mm)
|------|-----------|-------|-------------|-------------|-------------|--------|-------|---------|---------|-------
| [Meta](http://gvv.mpi-inf.mpg.de/projects/VNect/) |     87.7    |   77.4   |      74.7     |      72.9     |      51.3     |  83.3  |  80.1 |   76.7  |   40.4  | 124.7 
| [Meta](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) |     86.6    |   75.3   |      74.8     |      73.7     |      52.2     |  82.1  |  77.5 |   75.7  |   39.3  | 117.6
| Ours |       **90.4**    |   **79.1**   |     **88.5**    |      **81.6**     |      **66.3**     |  **91.9**  |  **92.2** |   **81.8**  |   **45.2**  | **89.4**

(Update) There has been a minor correction to the annotations for TS3 and TS4 in the test set. Results after correction:

|      | Stand/ Walk | Exercies | Sit on  Chair | Crouch/ Reach | On the  Floor | Sports | Misc. | All PCK | All AUC | MPJPE(mm)
|------|-----------|-------|-------------|-------------|-------------|--------|-------|---------|---------|
| Ours |      95.5    |   82.3   |     89.9   |     84.6    |     66.5     |  92.0  |  93.0 |  84.3 |   47.5  | 84.5 
