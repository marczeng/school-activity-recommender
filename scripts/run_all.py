# scripts/run_all.py
from __future__ import annotations
import sys
import subprocess   # 运行子进程：调用你已有的脚本
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（保证相对路径稳定）
SCRIPTS = [
    "scripts/clean_data.py",         # 若不存在会自动跳过
    "scripts/handle_outliers.py",
    "scripts/scale_encode.py",
    "scripts/train_model.py",
    "scripts/predict_batch.py",
]

def run_step(script_rel: str) -> bool:
    """
    运行单个子脚本：
      - 如果脚本不存在：提示并跳过（仅对 clean_data 这类可选步骤）
      - 否则用当前 Python 解释器执行，失败则返回 False
    """
    path = ROOT / script_rel
    if not path.exists():
        # 对“clean_data.py”等可选脚本友好跳过；其余脚本建议都在
        print(f"  Skip (not found): {script_rel}")
        return True

    print(f"▶  Running: {script_rel}")
    # subprocess.run：启动子进程；check=True 时非零退出会抛异常
    # sys.executable：使用当前解释器（避免用到别的 Python 环境）
    try:
        subprocess.run([sys.executable, "-u", str(path)], check=True, cwd=str(ROOT))
        print(f" Done: {script_rel}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Failed: {script_rel} (exit code {e.returncode})")
        return False

def main():
    t0 = perf_counter()  # 高精度计时
    print(f" Start pipeline at project root: {ROOT}\n")

    for script in SCRIPTS:
        ok = run_step(script)
        if not ok:
            print(" Pipeline stopped due to error.")
            break
    else:
        # 只有 for 循环全部成功才会进入这里（没有 break）
        dt = perf_counter() - t0
        print(f" Pipeline finished successfully in {dt:.2f}s")

if __name__ == "__main__":
    main()
