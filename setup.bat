@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: ==========================================
::   AI-Sentinel 一键环境配置脚本
::   适用于新电脑迁移，自动完成所有环境搭建
:: ==========================================

set "ENV_NAME=microtool-ai"
set "PYTHON_VER=3.11"
set "SCRIPT_DIR=%~dp0"

:: ------------------------------------------
:: 0. 颜色与样式函数
:: ------------------------------------------
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║     AI-Sentinel 一键环境配置                 ║
echo  ║     自动安装所有依赖，无需手动操作           ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ------------------------------------------
:: 1. 检查 Conda 是否可用
:: ------------------------------------------
echo [1/7] 检查 Conda 环境...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ╔══════════════════════════════════════════════╗
    echo  ║  [错误] 未检测到 Conda！                     ║
    echo  ║                                              ║
    echo  ║  请先安装 Miniconda 或 Anaconda:              ║
    echo  ║  https://docs.conda.io/en/latest/miniconda.html ║
    echo  ║                                              ║
    echo  ║  安装后请确保勾选 "Add to PATH" 选项         ║
    echo  ╚══════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

:: 获取 Conda 版本
for /f "tokens=2" %%v in ('conda --version 2^>nul') do set "CONDA_VER=%%v"
echo     √ Conda 已安装 (版本: %CONDA_VER%)

:: ------------------------------------------
:: 2. 检查或创建 Conda 环境
:: ------------------------------------------
echo.
echo [2/7] 检查 Conda 环境 "%ENV_NAME%"...

conda info --envs 2>nul | findstr /C:"%ENV_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo     √ 环境 "%ENV_NAME%" 已存在
    echo     - 将跳过创建步骤，直接安装/更新依赖
) else (
    echo     - 环境不存在，正在创建 (Python %PYTHON_VER%)...
    echo     - 这可能需要几分钟，请耐心等待...
    conda create -n %ENV_NAME% python=%PYTHON_VER% -y
    if !errorlevel! neq 0 (
        echo [错误] 创建 Conda 环境失败！
        pause
        exit /b 1
    )
    echo     √ 环境创建成功
)

:: ------------------------------------------
:: 3. 激活环境
:: ------------------------------------------
echo.
echo [3/7] 激活环境 "%ENV_NAME%"...
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo [错误] 激活环境失败！
    echo 请尝试运行: conda init cmd.exe
    echo 然后重新打开命令行窗口再运行此脚本
    pause
    exit /b 1
)
echo     √ 环境已激活

:: 显示 Python 版本
for /f "tokens=*" %%p in ('python --version 2^>nul') do echo     - %%p

:: ------------------------------------------
:: 4. 安装 PyTorch (GPU 优先)
:: ------------------------------------------
echo.
echo [4/7] 安装 PyTorch...

:: 检查是否已安装 torch
python -c "import torch; print(torch.__version__)" >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%t in ('python -c "import torch; print(torch.__version__)"') do (
        echo     √ PyTorch 已安装 (版本: %%t^)
    )
    python -c "import torch; print(torch.cuda.is_available())" 2>nul | findstr /C:"True" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%g in ('python -c "import torch; print(torch.cuda.get_device_name(0))"') do (
            echo     - GPU 可用: %%g
        )
    ) else (
        echo     - GPU 不可用，使用 CPU 模式
    )
) else (
    echo     - 检测 NVIDIA GPU...
    nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        echo     - 检测到 NVIDIA GPU，安装 CUDA 版 PyTorch...
        echo     - 这可能需要较长时间（约 2GB 下载），请耐心等待...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ) else (
        echo     - 未检测到 NVIDIA GPU，安装 CPU 版 PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
    if !errorlevel! neq 0 (
        echo [警告] PyTorch 安装可能不完整，后续可手动安装
    ) else (
        echo     √ PyTorch 安装成功
    )
)

:: ------------------------------------------
:: 5. 安装 PaddlePaddle (OCR 依赖)
:: ------------------------------------------
echo.
echo [5/7] 安装 PaddlePaddle (OCR 引擎)...

python -c "import paddle; print(paddle.__version__)" >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%p in ('python -c "import paddle; print(paddle.__version__)"') do (
        echo     √ PaddlePaddle 已安装 (版本: %%p^)
    )
) else (
    nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        echo     - 安装 GPU 版 PaddlePaddle...
        pip install paddlepaddle-gpu
    ) else (
        echo     - 安装 CPU 版 PaddlePaddle...
        pip install paddlepaddle
    )
    if !errorlevel! neq 0 (
        echo [警告] PaddlePaddle 安装可能不完整
    ) else (
        echo     √ PaddlePaddle 安装成功
    )
)

:: ------------------------------------------
:: 6. 安装其他 Python 依赖
:: ------------------------------------------
echo.
echo [6/7] 安装项目依赖...

:: 使用 requirements 文件安装，但跳过 torch 和 paddle（已单独处理）
pip install opencv-python>=4.8.0 numpy>=1.24.0 ^
    transformers>=4.36.0 accelerate>=0.25.0 ^
    qwen-vl-utils>=0.0.2 ^
    chromadb>=0.4.0 ^
    streamlit>=1.28.0 requests>=2.31.0 ^
    pyyaml>=6.0 Pillow>=10.0.0 ^
    paddleocr>=2.7.0 ^
    streamlit-cropper>=0.2.0 ^
    ultralytics

if %errorlevel% neq 0 (
    echo [警告] 部分依赖安装可能失败，请检查上方日志
) else (
    echo     √ 所有 Python 依赖安装完成
)

:: ------------------------------------------
:: 7. 检查外部工具
:: ------------------------------------------
echo.
echo [7/7] 检查外部工具...

:: 检查 FFmpeg
where ffmpeg >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%f in ('ffmpeg -version 2^>nul ^| findstr /C:"ffmpeg version"') do (
        echo     √ FFmpeg 已安装 (%%f^)
    )
) else (
    echo     ✗ FFmpeg 未安装！
    echo       RTSP 摄像头功能需要 FFmpeg
    echo       下载地址: https://www.gyan.dev/ffmpeg/builds/
    echo       下载后将 ffmpeg.exe 所在目录添加到系统 PATH
)

:: 检查 ffprobe
where ffprobe >nul 2>&1
if %errorlevel% equ 0 (
    echo     √ ffprobe 已安装
) else (
    echo     ✗ ffprobe 未安装 (通常随 FFmpeg 一起)
)

:: ------------------------------------------
:: 完成
:: ------------------------------------------
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║          环境配置完成！                      ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  启动方式:
echo    方式1: 双击 start.bat
echo    方式2: conda activate %ENV_NAME%
echo           streamlit run src/webui/app.py
echo.

:: ------------------------------------------
:: 环境验证
:: ------------------------------------------
echo  ── 环境验证 ──────────────────────────────────
python -c "import torch; v=torch.__version__; g='GPU ✓' if torch.cuda.is_available() else 'CPU'; print(f'  PyTorch:      {v} ({g})')" 2>nul || echo   PyTorch:      未安装
python -c "import paddle; print(f'  PaddlePaddle: {paddle.__version__}')" 2>nul || echo   PaddlePaddle: 未安装
python -c "import streamlit; print(f'  Streamlit:    {streamlit.__version__}')" 2>nul || echo   Streamlit:    未安装
python -c "import chromadb; print(f'  ChromaDB:     {chromadb.__version__}')" 2>nul || echo   ChromaDB:     未安装
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>nul || echo   Transformers: 未安装
python -c "import cv2; print(f'  OpenCV:       {cv2.__version__}')" 2>nul || echo   OpenCV:       未安装
echo  ────────────────────────────────────────────────
echo.

pause
