import subprocess
import time

for i in range(100):
    process = subprocess.Popen("python ../static_learning_on_memory.py", shell=True)
    process.wait()
    print(process.returncode)
    time.sleep(1)
