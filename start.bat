@echo off

where /q git
if errorlevel 1 (
    echo You must install Git to proceed: https://git-scm.com
    exit /b
)

where /q npm
if errorlevel 1 (
    echo You must install Node.js to proceed: https://nodejs.org
    exit /b
)

npm install
node start.js
