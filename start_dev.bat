@echo off
REM ============================================================
REM  N.E.K.O. Developer Mode launcher
REM
REM  *** THIS FILE IS PURE ASCII ON PURPOSE. ***
REM
REM  Why: cmd.exe reads .bat files in the console code page, and it
REM  reads them AHEAD in blocks. A UTF-8 file with Chinese inside
REM  gets decoded as GBK on a 936 console, so every Chinese line
REM  turns into garbage -- on 2026-10-10 this made the launcher fail
REM  with "'_buddy' is not recognized as an internal or external
REM  command". Putting chcp 65001 inside the file does NOT fix it:
REM  changing the code page mid-file invalidates cmd's read offset,
REM  and the following lines still get decoded with the old page.
REM
REM  So: no non-ASCII byte anywhere, and no chcp line at all.
REM  run_loop prints its Chinese countdown using the console's own
REM  code page, which displays correctly either way.
REM
REM  --title genshin is an ASCII alias resolved by run_loop to the
REM  real window title. See TITLE_ALIASES in run_loop.py.
REM ============================================================

cd /d D:\NEKO\N.E.K.O-main\N.E.K.O-main\plugin\plugins\genshin_game_buddy

echo.
echo ============================================================
echo  N.E.K.O. Developer Mode
echo ============================================================
echo.
echo  1) Opening the developer overlay...
echo  2) The engine starts in 15 seconds.
echo.
echo  *** SWITCH TO GENSHIN (fullscreen) NOW ***
echo      The window must be VISIBLE, not minimized -- a minimized
echo      window collapses its client area to 0x0 and no capture
echo      channel can read it.
echo.

set "NEKO_PY=D:\NEKO\N.E.K.O-main\N.E.K.O-main\.venv\Scripts\python.exe"

start "" "%NEKO_PY%" -m autopilot.tools.overlay --mode dev --event-rows 10

"%NEKO_PY%" -m autopilot.tools.run_loop --title genshin --live --delay 15

echo.
echo  Engine exited.
pause
