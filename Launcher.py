import subprocess
import time

for i in range(10):
    process = subprocess.Popen("python ../launch_turbodroid_simu_agent.py", shell=True)
    process.wait()
    print(process.returncode)
    time.sleep(1)
