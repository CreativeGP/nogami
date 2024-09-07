## data_zx_qc.py のバージョン
1. data_zx_qc.py  
generate -> spider + to_gh -> diagram  
diagram -> extract_circuit -> qc  
qc -> full_optimize -> qc1  
num_v; ２qubit 補正なし  
data -> pkl  
summary -> summary/1

2. data_zx_qc2.py  
generate -> full_reduce -> diagram  
diagram -> extract_circuit -> qc  
qc -> full_optimize -> qc1  
num_v; ２qubit 補正あり  
num_v = num_v - num_2qv (num_2qv を引くことで、相殺)  
data -> pkl2  
summary -> summary/2


3. data_zx_qc3.py
generate -> full_reduce -> diagram  
diagram -> extract_circuit -> qc  
qc -> full_optimize -> qc1  
num_v; ２qubit 補正あり  
num_v = - num_v + num_2qv (num_2qv を引くことで、相殺)  
data -> pkl3  
summary -> summary/3