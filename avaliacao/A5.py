import Orange 
import matplotlib.pyplot as plt

names = ["C4.5", "C4.5+m", "C4.5+cf", "C4.5+m+cf"]
    
avranks = [3.143,2.000,2.893,1.964]
cd = Orange.evaluation.compute_CD(avranks, 14,alpha="0.05", test="bonferroni-dunn") #tested on 14 datasets 