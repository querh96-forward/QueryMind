@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "ACTION=%~1"
if "%ACTION%"=="" goto MENU
if /i "%ACTION%"=="start" goto START
if /i "%ACTION%"=="stop" goto STOP
if /i "%ACTION%"=="restart" goto RESTART
if /i "%ACTION%"=="status" goto STATUS
if /i "%ACTION%"=="logs" goto LOGS
if /i "%ACTION%"=="test" goto TEST
if /i "%ACTION%"=="eval" goto EVAL
if /i "%ACTION%"=="reset" goto RESET
echo 未知命令：%ACTION%
echo 用法：querymind.bat [start^|stop^|restart^|status^|logs^|test^|eval^|reset]
exit /b 1

:MENU
cls
echo ========================================
echo QueryMind 项目管理
echo ========================================
echo 1. 启动或更新服务
echo 2. 查看服务状态
echo 3. 查看实时日志
echo 4. 重启应用服务
echo 5. 运行自动化测试
echo 6. 运行离线评测
echo 7. 停止服务
echo 8. 清空并重建数据
echo 9. 退出
echo.
choice /c 123456789 /n /m "请选择："
if errorlevel 9 exit /b 0
if errorlevel 8 goto RESET
if errorlevel 7 goto STOP
if errorlevel 6 goto EVAL
if errorlevel 5 goto TEST
if errorlevel 4 goto RESTART
if errorlevel 3 goto LOGS
if errorlevel 2 goto STATUS
if errorlevel 1 goto START

:CHECK_DOCKER
where docker >nul 2>nul || (
  echo 未检测到 Docker。请先安装并启动 Docker Desktop。
  exit /b 1
)
docker info >nul 2>nul || (
  echo Docker Desktop 尚未就绪，请启动后重试。
  exit /b 1
)
exit /b 0

:START
call :CHECK_DOCKER || goto END_ERROR
if not exist .env (
  copy .env.example .env >nul
  echo 已创建 .env，请先填写 LLM_API_KEY、LLM_BASE_URL 和 EMBEDDING_API_KEY、EMBEDDING_BASE_URL。
)
echo 正在构建并启动 QueryMind...
docker compose up -d --build || goto END_ERROR
echo 正在等待健康检查...
for /l %%i in (1,1,60) do (
  curl.exe -fsS http://127.0.0.1:6006/api/v1/health 2>nul | findstr /C:"\"status\":\"ok\"" >nul && goto READY
  timeout /t 2 /nobreak >nul
)
echo 系统尚未就绪，请运行 querymind.bat logs 查看日志。
goto END_ERROR
:READY
echo QueryMind 已启动：http://127.0.0.1:6006
start "" http://127.0.0.1:6006
goto END_OK

:STOP
call :CHECK_DOCKER || goto END_ERROR
docker compose down
echo 服务已停止，PostgreSQL 与 Redis 数据卷会保留。
goto END_OK

:RESTART
call :CHECK_DOCKER || goto END_ERROR
docker compose restart api worker web-mcp querymind-mcp
echo 应用服务已重启。
goto END_OK

:STATUS
call :CHECK_DOCKER || goto END_ERROR
docker compose ps -a
goto END_OK

:LOGS
call :CHECK_DOCKER || goto END_ERROR
docker compose logs -f --tail=200
goto END_OK

:TEST
call :CHECK_DOCKER || goto END_ERROR
docker compose exec api pytest -q tests
goto END_OK

:EVAL
call :CHECK_DOCKER || goto END_ERROR
docker compose exec api python evals/run_eval.py
goto END_OK

:RESET
call :CHECK_DOCKER || goto END_ERROR
echo 此操作会删除对话、运行记录、记忆、评测结果和数据库数据。
set /p "CONFIRM=输入 RESET 确认："
if /i not "%CONFIRM%"=="RESET" (
  echo 已取消。
  goto END_OK
)
docker compose down -v
echo 数据已清空。下次启动时会重新导入 Northwind。
goto END_OK

:END_ERROR
echo 操作失败。
if "%~1"=="" pause
exit /b 1

:END_OK
if "%~1"=="" pause
exit /b 0
