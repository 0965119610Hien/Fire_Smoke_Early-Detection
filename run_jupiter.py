import modal
import subprocess

app = modal.App("fire-smoke-jupyter")
volume = modal.Volume.from_name("fire_smoke_dataset")

image = modal.Image.debian_slim().pip_install(
    "jupyter", "torch", "torchvision", "opencv-python-headless", "einops"
)

@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=86400)
def run_jupyter():
    # Sử dụng modal.forward để tạo đường dẫn Public ra ngoài
    with modal.forward(8888) as tunnel:
        print("\n" + "="*60)
        print(f"🚀 LINK TRUY CẬP JUPYTER CỦA BẠN ĐÂY: {tunnel.url}")
        print("="*60 + "\n")
        
        subprocess.run([
            "jupyter", "notebook", 
            "--ip=0.0.0.0", 
            "--port=8888", 
            "--no-browser", 
            "--allow-root", 
            "--NotebookApp.token=''" # Không dùng password
        ])