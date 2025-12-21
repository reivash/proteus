@echo off
REM Agent Owl launcher for Proteus market predictor research
REM This monitors the Claude Code terminal and prompts continuation when idle

echo ============================================================
echo   Starting Agent Owl for Proteus Market Predictor Research
echo ============================================================
echo.
echo Available configs:
echo   1) proteus_market_predictor.json - Market predictor logic research (DEFAULT)
echo   2) proteus.json                  - General research/improvements
echo   3) proteus_ipo_niche.json        - IPO niche market investigation
echo   4) proteus_codex.json            - Codex integration mode
echo.

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=proteus_market_predictor.json

echo Using config: %CONFIG%
echo.

cd /d "%~dp0..\agent-owl"
python agent_owl.py --config "configs/%CONFIG%"
