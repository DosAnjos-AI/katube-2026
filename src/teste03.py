# ~/katube_teste/katube-novo/src/teste_memory_monitor.py
import torch
import time
import subprocess
from pathlib import Path

def monitor_gpu_memory(interval=2, duration=300):
    """
    Monitora uso de memória GPU durante execução da pipeline.
    
    Args:
        interval: Segundos entre checks
        duration: Duração total do monitoramento (segundos)
    """
    
    log_file = Path("gpu_memory_log.txt")
    
    print(f"🔍 Iniciando monitoramento GPU")
    print(f"⏱️ Interval: {interval}s, Duration: {duration}s")
    print(f"📁 Log: {log_file}")
    print("=" * 60)
    
    with open(log_file, 'w') as f:
        f.write("timestamp,memory_allocated_mb,memory_reserved_mb,utilization%\n")
        
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            try:
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                    
                    # Get GPU utilization via nvidia-smi
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True,
                        text=True
                    )
                    gpu_util = result.stdout.strip()
                    
                    timestamp = time.strftime("%H:%M:%S")
                    log_line = f"{timestamp},{memory_allocated:.2f},{memory_reserved:.2f},{gpu_util}\n"
                    
                    f.write(log_line)
                    f.flush()
                    
                    print(f"[{timestamp}] Alloc: {memory_allocated:.2f}MB | Reserved: {memory_reserved:.2f}MB | Util: {gpu_util}%")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n⚠️ Monitoramento interrompido pelo usuário")
                break
            except Exception as e:
                print(f"❌ Erro no monitoramento: {e}")
                break
    
    print(f"✅ Log salvo em: {log_file}")

if __name__ == "__main__":
    monitor_gpu_memory()