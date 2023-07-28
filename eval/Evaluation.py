import os
print("test")
DRL_dir = "../solutionsDRL/"
import re
import os 
import fire
def extract_benchmark_num(DRL_dir_):
    files = os.listdir(DRL_dir_)
    filename_list = []
    for filename in files:
        if "benchmark" in filename:
            num = re.findall(r'\d+', filename)[0]
            filename_list.append(int(num))
    filename_list = sorted(filename_list)
    return filename_list
def main(folder = "test_data_",
         sol_dir = "solutionsDRL"
         ):
    #Nstep only miss [9,11] , [1,2,3,4,5,6,7,8,  10,   12,13,14,15,16,17,18,19,20] 
    #Nstep with pretrain        [2,3,4,5,    8,  10,11,12,13,14,      17,18,   20] winning rate 0.69
    #                           [1, 2, 4, 5, 6, 10, 11, 12, 13, 17]
    #noisy  miss alot  ,        [2,3,  5,6,  8,  10,  12,13,14,15,16,      19,20] #*but winning 
    # DTQN is good for pretraining, only miss 2[  2,3,4,5,6,7,8,   10,11,12,13,14,15,16,17,18,19,20]
    # rate very high  0.77
    # DTQN pretrain PER [1,2,3,4,5,   7,8,   10,11,12,13,14,15,16,17,18,19,20]
    # DTQN PER train and save [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20]
    sol_dir = f"../{sol_dir}/"
    tb_dir = f"../{folder}/benchmark_reduced/"
    Astar_dir = f"../{folder}/solutionsA*_reduced/"
    os.system(f"rm -r report.txt")
    benchmark_num = extract_benchmark_num(sol_dir)
    
    for b in benchmark_num:
        print("========== benchmark ->  DRL solution   =========")
        os.system(f"perl eval2008.pl {tb_dir}test_benchmark_{b}.gr\
        {sol_dir}test_benchmark_{b}.gr.DRLsolution")        #command

        print("========== benchmark -> Astar solution  =========")
        os.system(f"perl eval2008.pl {tb_dir}test_benchmark_{b}.gr\
         {Astar_dir}test_benchmark_{b}.grAstar_solution")     #command
    print(benchmark_num)
# target = extract_benchmark_num(DRL_dir)
if __name__ == "__main__":
    fire.Fire(main)



