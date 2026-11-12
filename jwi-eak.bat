@echo off
rem Jwi-eak launcher: jwi-eak.bat [FILE] [options]   (jwi-eak.bat --help)
julia --project="%~dp0." -t auto "%~dp0Jwi-eak.jl" %*
