# K-Means++ & VQ_LBG
This is the repo for ***Pattern Recognition Course Task 1***, an implement of K-Means++ and VQ_LBG algorithms written in 
Python. All codes were tested in Ubuntu 18.04.   
![example](https://github.com/HoraceKem/clustering/blob/master/pics/example.png)
## Preparation
 Installing dependency packages may change your packages' version, so it would be a good choice to create a virtual 
 environment.   
```
conda create -n PR_Task1 python=3.7
conda activate PR_Task1
pip install -r requirements.txt
```
## Running
Change to folder ```sources``` and command:   
```
python main.py <mode> [<options>] 
``` 
```<mode>``` should be one of the following:   
1. ```KM``` -Use K-Means to cluster the data.
2. ```KMPP```-Use K-Means++ to cluster the data.
3. ```VQ_LBG```-Use VQ_LBG to cluser the data.
4. ```ALL```-Run the upper three algorithms in order and compare the results.   

```<options>``` have default values or you can change it by yourself:   
+ ```-visualize``` **[bool, True]** Visualize the result using matplotlib
+ ```-cluster_num``` **[int, 4]** 
+ ```-sample_num``` **[int, 100]**
+ ```-center``` **[str, "2,2;8,2;5,8;4,4"]** You can also set this argument as 'auto' to produce center points.
+ ```-save_name``` **[str, None]** The filename of the generated data, value 'None' will drop the data after running.
+ ```-use_saved_data``` **[str, None]** Use saved data to ensure the fairness of each test

## Contact
Horace.Kem, Soochow University   
E-mail:horacekem@163.com   
[GitHub](github.com/horacekem)
