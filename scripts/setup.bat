@echo off
setlocal EnableDelayedExpansion

REM Change to file root, even if you run it elsewhere
cd /d "%~dp0.."

REM Verify Python is installed, just a basic guard clause
python --version >nul 2>&1
if errorlevel 1 (
	echo Python is not installed or not in PATH.
	pause
	exit /b 1
)

REM Check Python version >= 3.10
REm Some ugly python code, but way cleaner than batch here
python -c ^
"import sys; ^
print(f'Detected: {sys.version.split()[0]}'); ^
sys.exit(0 if sys.version_info ^>= (3, 10) else 1)^" || (
	echo.
	echo [ERROR] Python 3.10 or higher is required.
	pause
	exit /b 1
)
echo Python version detected and verified.


REM Create venv only if it doesn't already exist
if exist ".venv\Scripts\python.exe" (
	echo Virtual environment already exists, skipping creation.
) else (
	echo Creating virtual environment...
	python -m venv .venv
	if errorlevel 1 (
		echo Failed to create virtual environment.
		pause
		exit /b 1
	)
)

REM Detect NVIDIA GPU. We will reinstall torch with the matching CUDA build AFTER
REM the main requirements install, so whisperx's torch version pin does not overwrite
REM the CUDA build with a CPU-only wheel from PyPI.
echo Checking for NVIDIA GPU...
set HAS_GPU=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
	set HAS_GPU=1
	REM Determine CUDA version from nvidia-smi output (e.g. "CUDA Version: 12.4").
	REM We map to the closest PyTorch CUDA wheel: cu118, cu121, or cu124.
	for /f "tokens=*" %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set DRIVER=%%v
	for /f "tokens=9" %%c in ('nvidia-smi ^| findstr /i "CUDA Version"') do set CUDA_VER=%%c

	REM Default to cu121 if we cannot parse the version.
	set TORCH_INDEX=https://download.pytorch.org/whl/cu121
	if defined CUDA_VER (
		for /f "tokens=1 delims=." %%M in ("!CUDA_VER!") do set CUDA_MAJOR=%%M
		for /f "tokens=2 delims=." %%m in ("!CUDA_VER!") do set CUDA_MINOR=%%m
		REM CUDA < 12      ->  PyTorch 2.8.0 dropped CUDA 11.x support; skip CUDA reinstall
		REM CUDA 12.0-12.3 ->  cu121
		REM CUDA 12.4-12.7 ->  cu124
		REM CUDA 12.8+     ->  cu128  (also used for CUDA 13+ until a cu130 wheel exists)
		if !CUDA_MAJOR! LSS 12 (
			echo.
			echo WARNING: Your NVIDIA driver reports CUDA !CUDA_VER!, but PyTorch 2.8.0 requires CUDA 12.1+.
			echo GPU acceleration is unavailable with your current drivers.
			echo To enable it, update your NVIDIA drivers from https://www.nvidia.com/drivers
			echo Keeping CPU-only PyTorch 2.8.0.
			echo.
			set HAS_GPU=0
		) else if !CUDA_MAJOR! GTR 12 (
			set TORCH_INDEX=https://download.pytorch.org/whl/cu128
		) else (
			if !CUDA_MINOR! GEQ 8 (
				set TORCH_INDEX=https://download.pytorch.org/whl/cu128
			) else if !CUDA_MINOR! GEQ 4 (
				set TORCH_INDEX=https://download.pytorch.org/whl/cu124
			)
		)
	)
	if !HAS_GPU! == 1 (
		echo NVIDIA GPU detected. Will install CUDA-enabled PyTorch from !TORCH_INDEX! after requirements.
	)
) else (
	echo No NVIDIA GPU detected. PyTorch will be installed as CPU-only.
)

REM Install all requirements first. This lets pip resolve version constraints correctly
REM (e.g. whisperx requires torch~=2.8.0) before we swap in the CUDA build.
echo Installing requirements...
.venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
	echo Failed to install requirements.
	pause
	exit /b 1
)

REM Now reinstall torch + torchaudio with the CUDA build, replacing the CPU wheels
REM that were pulled in by the dependency resolver above.
REM --force-reinstall: replaces same-version CPU wheel with CUDA wheel
REM --no-deps: skips re-resolving deps to avoid another CPU torch being pulled back in
if !HAS_GPU! == 1 (
	echo Reinstalling PyTorch with CUDA support from !TORCH_INDEX!...
	.venv\Scripts\pip.exe install torch torchaudio --index-url !TORCH_INDEX! --force-reinstall --no-deps
	if errorlevel 1 (
		echo Warning: CUDA PyTorch reinstall failed. CPU-only torch remains installed.
	) else (
		echo CUDA-enabled PyTorch installed successfully.
	)
)

REM Copy .env.example if .env doesn't exist
if not exist ".env" (
	if exist "config\.env.example" (
		echo Copying config\.env.example to .env...
		copy /y "config\.env.example" ".env" >nul
	) else (
		echo config\.env.example not found, could not create .env.
	)
) else (
	echo .env already exists.
)

echo.
echo Setup complete!
echo To start the web server, run:
echo .venv\Scripts\activate
echo uvicorn Web.app:app --reload
echo Or launch the scripts/StartWebserver.bat
echo Change the portor API keys in the .env file, 
pause

endlocal
