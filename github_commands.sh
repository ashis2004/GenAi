#!/bin/bash
# GitHub Quick Commands for GenAI Repository
# This script provides quick commands for common Git operations

echo ""
echo "================================================"
echo "   GenAI Repository - GitHub Quick Commands"
echo "================================================"
echo ""

show_menu() {
    echo "Choose an option:"
    echo ""
    echo "1. Check repository status"
    echo "2. Quick commit and push"
    echo "3. Create new feature branch"
    echo "4. Sync with GitHub (pull latest)"
    echo "5. Push current branch"
    echo "6. Switch to main branch"
    echo "7. View commit history"
    echo "8. Setup repository (first time)"
    echo "9. Open GitHub repository in browser"
    echo "0. Exit"
    echo ""
}

check_status() {
    echo ""
    echo "==========[ Repository Status ]=========="
    git status
    echo ""
    git branch
    echo ""
    read -p "Press Enter to continue..."
}

quick_commit() {
    echo ""
    read -p "Enter commit message: " message
    
    if [ -z "$message" ]; then
        echo "No message provided. Cancelling."
        read -p "Press Enter to continue..."
        return
    fi
    
    echo ""
    echo "Adding all changes..."
    git add .
    
    echo "Committing changes..."
    git commit -m "$message"
    
    echo "Pushing to GitHub..."
    git push
    
    echo ""
    echo "Done! Changes committed and pushed."
    read -p "Press Enter to continue..."
}

new_branch() {
    echo ""
    read -p "Enter branch name (e.g., add-new-feature): " branch_name
    
    if [ -z "$branch_name" ]; then
        echo "No branch name provided. Cancelling."
        read -p "Press Enter to continue..."
        return
    fi
    
    echo ""
    echo "Switching to main branch..."
    git checkout main
    
    echo "Pulling latest changes..."
    git pull origin main
    
    echo "Creating new branch: feature/$branch_name"
    git checkout -b "feature/$branch_name"
    
    echo ""
    echo "Branch created! You're now on feature/$branch_name"
    read -p "Press Enter to continue..."
}

sync_repo() {
    echo ""
    echo "==========[ Syncing with GitHub ]=========="
    echo "Fetching latest changes..."
    git fetch origin
    
    echo "Switching to main branch..."
    git checkout main
    
    echo "Pulling latest changes..."
    git pull origin main
    
    echo ""
    echo "Repository synced with GitHub!"
    read -p "Press Enter to continue..."
}

push_branch() {
    echo ""
    echo "==========[ Pushing Current Branch ]=========="
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"
    
    echo "Pushing $current_branch to GitHub..."
    git push origin "$current_branch"
    
    echo ""
    echo "Branch pushed! If this is a feature branch, don't forget to create a Pull Request."
    read -p "Press Enter to continue..."
}

main_branch() {
    echo ""
    echo "Switching to main branch..."
    git checkout main
    
    echo "Pulling latest changes..."
    git pull origin main
    
    echo ""
    echo "Now on main branch with latest changes."
    read -p "Press Enter to continue..."
}

show_history() {
    echo ""
    echo "==========[ Recent Commit History ]=========="
    git log --oneline -10
    echo ""
    read -p "Press Enter to continue..."
}

setup_repo() {
    echo ""
    echo "==========[ First Time Setup ]=========="
    echo ""
    
    echo "Step 1: Initialize Git repository..."
    git init
    
    echo "Step 2: Add all files..."
    git add .
    
    echo "Step 3: Create initial commit..."
    git commit -m "Initial commit: Complete GenAI implementation"
    
    echo "Step 4: Set main branch..."
    git branch -M main
    
    echo "Step 5: Add GitHub remote..."
    git remote add origin https://github.com/ashis2004/GenAi.git
    
    echo "Step 6: Push to GitHub..."
    git push -u origin main
    
    echo ""
    echo "Setup complete! Your repository is now on GitHub."
    echo "Make sure you've created the repository on GitHub first."
    read -p "Press Enter to continue..."
}

open_github() {
    echo ""
    echo "Opening GitHub repository in browser..."
    
    # Detect OS and open browser accordingly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open https://github.com/ashis2004/GenAi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open https://github.com/ashis2004/GenAi
    else
        echo "Please open: https://github.com/ashis2004/GenAi"
    fi
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (0-9): " choice
    
    case $choice in
        0)
            echo ""
            echo "Goodbye! Happy coding with GenAI! 🚀"
            echo ""
            exit 0
            ;;
        1)
            check_status
            ;;
        2)
            quick_commit
            ;;
        3)
            new_branch
            ;;
        4)
            sync_repo
            ;;
        5)
            push_branch
            ;;
        6)
            main_branch
            ;;
        7)
            show_history
            ;;
        8)
            setup_repo
            ;;
        9)
            open_github
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done
