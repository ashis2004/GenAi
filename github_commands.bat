PS C:\21MIP\LLM\GitHub> git init
git : The term 'git' is not recognized as the name of a cmdlet, function, script file, or operable program. Check 
the spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ git init
+ ~~~
    + CategoryInfo          : ObjectNotFound: (git:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException

PS C:\21MIP\LLM\GitHub> cd GenAi
PS C:\21MIP\LLM\GitHub\GenAi> git init
git : The term 'git' is not recognized as the name of a cmdlet, function, script file, or operable program. Check 
the spelling of the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ git init
+ ~~~
    + CategoryInfo          : ObjectNotFound: (git:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException@echo off
REM GitHub Quick Commands for GenAI Repository
REM This batch file provides quick commands for common Git operations

echo.
echo ================================================
echo    GenAI Repository - GitHub Quick Commands
echo ================================================
echo.

:menu
echo Choose an option:
echo.
echo 1. Check repository status
echo 2. Quick commit and push
echo 3. Create new feature branch
echo 4. Sync with GitHub (pull latest)
echo 5. Push current branch
echo 6. Switch to main branch
echo 7. View commit history
echo 8. Setup repository (first time)
echo 9. Open GitHub repository in browser
echo 0. Exit
echo.
set /p choice="Enter your choice (0-9): "

if "%choice%"=="0" goto exit
if "%choice%"=="1" goto status
if "%choice%"=="2" goto quick_commit
if "%choice%"=="3" goto new_branch
if "%choice%"=="4" goto sync
if "%choice%"=="5" goto push_branch
if "%choice%"=="6" goto main_branch
if "%choice%"=="7" goto history
if "%choice%"=="8" goto setup
if "%choice%"=="9" goto open_github

echo Invalid choice. Please try again.
goto menu

:status
echo.
echo ==========[ Repository Status ]==========
git status
echo.
git branch
echo.
pause
goto menu

:quick_commit
echo.
set /p message="Enter commit message: "
if "%message%"=="" (
    echo No message provided. Cancelling.
    pause
    goto menu
)

echo.
echo Adding all changes...
git add .

echo Committing changes...
git commit -m "%message%"

echo Pushing to GitHub...
git push

echo.
echo Done! Changes committed and pushed.
pause
goto menu

:new_branch
echo.
set /p branch_name="Enter branch name (e.g., add-new-feature): "
if "%branch_name%"=="" (
    echo No branch name provided. Cancelling.
    pause
    goto menu
)

echo.
echo Switching to main branch...
git checkout main

echo Pulling latest changes...
git pull origin main

echo Creating new branch: feature/%branch_name%
git checkout -b feature/%branch_name%

echo.
echo Branch created! You're now on feature/%branch_name%
pause
goto menu

:sync
echo.
echo ==========[ Syncing with GitHub ]==========
echo Fetching latest changes...
git fetch origin

echo Switching to main branch...
git checkout main

echo Pulling latest changes...
git pull origin main

echo.
echo Repository synced with GitHub!
pause
goto menu

:push_branch
echo.
echo ==========[ Pushing Current Branch ]==========
for /f "tokens=*" %%i in ('git branch --show-current') do set current_branch=%%i
echo Current branch: %current_branch%

echo Pushing %current_branch% to GitHub...
git push origin %current_branch%

echo.
echo Branch pushed! If this is a feature branch, don't forget to create a Pull Request.
pause
goto menu

:main_branch
echo.
echo Switching to main branch...
git checkout main

echo Pulling latest changes...
git pull origin main

echo.
echo Now on main branch with latest changes.
pause
goto menu

:history
echo.
echo ==========[ Recent Commit History ]==========
git log --oneline -10
echo.
pause
goto menu

:setup
echo.
echo ==========[ First Time Setup ]==========
echo.
echo Step 1: Initialize Git repository...
git init

echo Step 2: Add all files...
git add .

echo Step 3: Create initial commit...
git commit -m "Initial commit: Complete GenAI implementation"

echo Step 4: Set main branch...
git branch -M main

echo Step 5: Add GitHub remote...
git remote add origin https://github.com/ashis2004/GenAi.git

echo Step 6: Push to GitHub...
git push -u origin main

echo.
echo Setup complete! Your repository is now on GitHub.
echo Make sure you've created the repository on GitHub first.
pause
goto menu

:open_github
echo.
echo Opening GitHub repository in browser...
start https://github.com/ashis2004/GenAi
goto menu

:exit
echo.
echo Goodbye! Happy coding with GenAI! 🚀
echo.
pause
