@echo off
REM ============================================================
REM  Window-find diagnostic launcher
REM
REM  *** PURE ASCII ON PURPOSE ***  (see start_dev.bat for why)
REM
REM  Run this while Genshin Impact is VISIBLE (not minimized).
REM  It prints three things side by side:
REM    A) what find_genshin_window() sees from a normal process
REM    B) what the title-only fallback sees
REM    C) what the PLUGIN HOST process reported in its own log
REM ============================================================

cd /d D:\NEKO\N.E.K.O-main\N.E.K.O-main\plugin\plugins\genshin_game_buddy

echo.
echo  *** SWITCH TO GENSHIN (fullscreen) NOW ***
echo      The window must be VISIBLE, not minimized -- a
echo      minimized window collapses its client area to 0x0
echo      and no capture channel can read it.
echo      Probing in 15 seconds...
echo.

REM countdown. "timeout" is the clean one, but it REFUSES to run when
REM stdin is not a console and exits non-zero -- so fall back to a
REM 1-second loopback ping, which always works.
for /L %%i in (15,-1,1) do (
    echo    %%i seconds left...
    timeout /t 1 /nobreak >nul 2>&1 || ping -n 2 127.0.0.1 >nul
)

echo.
"D:\NEKO\N.E.K.O-main\N.E.K.O-main\.venv\Scripts\python.exe" -m autopilot.tools.window_find_probe

echo.
pause
