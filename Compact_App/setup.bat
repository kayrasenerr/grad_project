call python -m venv .venv || exit /b !ERRORLEVEL!
call .venv\Scripts\activate || exit /b !ERRORLEVEL!
call python -m pip install -r requirements.txt || exit /b !ERRORLEVEL!
echo Installation completed
pause