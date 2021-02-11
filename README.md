# LiPPA

The code to reproduce the results of LiPPA. 

## Approximation of the number of linear regions

To reproduce the results that are presented in Figure 4 and Table 1, you have to run:



      python mlp_linregcount.py model_pth_path csv_file_path EPS_H



with :

model_pth_path, the path to the mlp_b model,
    
csv_file_path, the path to the csv file that will contains the information to exploit,
    
EPS_H, the value of epsilon, the $\ell_\infty$ ball around the original
    examples where linear regions are counted.
   
## LiPPA performance

To reproduce the results in Table II, you have to run : 

      python main_lippa_$architecture$.py model_pth_path csv_file_path EPS_ATTACK
   
with :
    
model_pth_path, the path to the mlp_b model,


csv_file_path, the path to the csv file that will contains the information to exploit,


EPS_H, the value of epsilon, the allowed $\ell_\infty$ deformation around original examples.
   
For example : python main_lippa_mlp_b.py pth_folder/robust_verify_benchmark/NOR_MLP_B.pth "nor_mlp_b_0.05.csv" 0.05.



