import os
import subprocess
import statistics

os.system("make clean && make VECTORIZE=1 AVX2=1")
cnt = 0
run_sec = []
while cnt < 50:
    out = subprocess.run(['srun', './test_auto_vectorize', '-t', '1'], capture_output=True)
    if out.returncode == 0:
        cnt += 1
        out = out.stdout.decode('utf-8')
        out = out.split('\n')[-2]
        out = out.split('sec')[0]
        run_sec.append(float(out))
print(statistics.median(run_sec))
