import subprocess
import sys
import time

# Reconfigure stdout to support UTF-8 diacritics in Windows console
sys.stdout.reconfigure(encoding='utf-8')

cmd = [
    r"C:\Users\Admin\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
    "-interaction=nonstopmode",
    "timesformer_report.tex"
]

print("Starting compilation...")
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace'
)

# Đọc output theo thời gian thực
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())
        sys.stdout.flush()

rc = process.poll()
print(f"Compilation finished with return code: {rc}")
if rc == 0:
    print("SUCCESS: PDF compiled successfully.")
else:
    print("ERROR: Compilation failed. Please check the logs above for errors.")
