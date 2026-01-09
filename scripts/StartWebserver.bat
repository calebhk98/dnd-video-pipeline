@echo off
setlocal
REM Change to file root, even if you run it elsewhere
cd /d "%~dp0.."

rem Grabbing the port, or defaulting to 8500
set "PORT=8500"
set "NEEDS_SETUP=0"

if not exist ".env" (
	set "NEEDS_SETUP=1"
) 
if not exist ".venv\Scripts\python.exe" (
	set "NEEDS_SETUP=1"
) 


if "%NEEDS_SETUP%"=="1" (
	rem No .env means the setup was not run
	rem run setup
	echo Seting up environment
	call Scripts/setup.bat

	if not exist ".env" (
		echo Install failed, closing
		pause
		exit /b 1
	) 
	if not exist ".venv\Scripts\python.exe" (
		echo Install failed, closing
		pause
		exit /b 1
	) 
) 

echo Setting up the variables
for /f "tokens=2 delims==" %%a in ('findstr /I /B "local_Server_Port" .env') do (
		set "PORT=%%a"
)

REM Strip any accidental spaces
set "PORT=%PORT: =%"

rem setup the .venv env
echo Activating the environment
call .venv\Scripts\activate

rem launch the server
echo Launching the Server
echo Auto launching to go to http://localhost:%PORT%
echo press Ctrl+C to close

start http://localhost:%PORT%
set "PYTHONPATH=%CD%"
cd Web
uvicorn app:app --reload --port %PORT%


pause
endlocal
