import subprocess
import sys
import os
import json

def setup_environment():
    """设置环境变量确保UTF-8编码"""
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    
    # Windows特定设置
    if sys.platform == 'win32':
        os.environ['PYTHONLEGACYWINDOWSFSENCODING'] = '0'
        os.environ['PYTHONUTF8'] = '1'

def check_pytorch_installation():
    """检查PyTorch安装和基本功能"""
    print("\n🩺 正在检查PyTorch环境...")
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        
        # 尝试一个简单的PyTorch操作
        x = torch.rand(5, 3)
        print(f"✅ PyTorch基本操作成功 (创建了一个 {x.shape} 的张量)")
        
        # 检查CUDA可用性 (如果安装了CUDA版本的PyTorch)
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用. CUDA版本: {torch.version.cuda}")
            print(f"   检测到 {torch.cuda.device_count()} 个CUDA设备.")
            print(f"   当前CUDA设备: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
        else:
            print("ℹ️ CUDA 不可用 (或者您安装的是CPU版本的PyTorch).")
        return True
    except ImportError:
        print("❌ PyTorch 未安装. 请根据您的环境安装PyTorch (CPU或CUDA版本).")
        print("   CPU版本示例: pip install torch torchvision torchaudio")
        print("   或通过conda: conda install pytorch torchvision torchaudio cpuonly -c pytorch")
        return False
    except Exception as e:
        print(f"❌ PyTorch 环境检查失败: {e}")
        print("   请确保PyTorch已正确安装并能正常工作.")
        return False

def check_requirements():
    """检查其他核心依赖"""
    print("\n📦 正在检查其他核心依赖...")
    try:
        import streamlit
        import sentence_transformers
        import faiss
        import numpy
        import pandas
        import requests
        print("✅ 其他核心依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少核心依赖: {e}")
        print("   请运行: pip install -r requirements.txt")
        return False

def ensure_data_directory():
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✅ data目录已创建/确认存在")
    
    # 检查数据文件，如果不存在则创建空的JSON文件
    for filename in ['laws.json', 'cases.json']:
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            print(f"⚠️  {filepath} 文件不存在，将创建一个空的JSON列表文件.")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([], f)
        else:
            print(f"✅ {filepath} 文件已存在.")

def main():
    print("="*60)
    print("🏛️  法律领域RAG系统 - 启动与环境诊断程序")
    print("="*60)
    
    setup_environment()
    print("✅ 环境变量已设置为UTF-8优先")
    
    if not check_pytorch_installation():
        print("\n❌ PyTorch环境存在问题，请先解决上述错误后再运行。")
        sys.exit(1)
        
    if not check_requirements():
        print("\n❌ 核心依赖缺失，请先解决上述错误后再运行。")
        sys.exit(1)
    
    ensure_data_directory()
    
    print("\n🚀 环境检查通过，准备启动Streamlit应用...")
    print("   请在浏览器中访问: http://localhost:8501")
    print("   按 Ctrl+C 停止应用")
    print("="*60)
    
    try:
        env = os.environ.copy()
        # 再次确保子进程使用UTF-8
        env['PYTHONIOENCODING'] = 'utf-8'
        if sys.platform == 'win32':
            env['PYTHONUTF8'] = '1'
            
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py", 
            "--server.port=8501"
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 应用已停止.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit应用启动失败: {e}")
        print("   请检查app.py是否存在错误.")
    except FileNotFoundError:
        print("❌ 无法找到streamlit命令. Streamlit是否已正确安装并添加到PATH?")

if __name__ == "__main__":
    main() 