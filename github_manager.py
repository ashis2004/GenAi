#!/usr/bin/env python3
"""
GitHub Repository Management Helper Script

This script helps you set up and manage your GenAI repository on GitHub.
It provides interactive commands for common Git operations.
"""

import json
import os
import subprocess
import sys
from datetime import datetime


class GitHubManager:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        self.repo_name = "GenAi"
        self.github_username = "ashis2004"  # Change this to your username
        
    def run_command(self, command, description=""):
        """Run a git command and return the result."""
        print(f"\n🔄 {description or command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.repo_path
            )
            if result.stdout.strip():
                print(f"✅ {result.stdout.strip()}")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print(f"Details: {e.stderr}")
            return False, e.stderr

    def check_git_status(self):
        """Check current git status."""
        print("\n" + "="*50)
        print("📊 CURRENT REPOSITORY STATUS")
        print("="*50)
        
        # Check if git is initialized
        if not os.path.exists(os.path.join(self.repo_path, '.git')):
            print("❌ Git repository not initialized")
            return False
        
        # Get current branch
        success, branch = self.run_command("git branch --show-current", "Current branch")
        
        # Get status
        success, status = self.run_command("git status --short", "File status")
        
        # Get last commit
        success, last_commit = self.run_command(
            "git log -1 --oneline", 
            "Last commit"
        )
        
        # Get remote info
        success, remote = self.run_command("git remote -v", "Remote repositories")
        
        return True

    def initialize_repository(self):
        """Initialize git repository if not already done."""
        print("\n" + "="*50)
        print("🚀 INITIALIZING GIT REPOSITORY")
        print("="*50)
        
        if os.path.exists(os.path.join(self.repo_path, '.git')):
            print("✅ Git repository already initialized")
            return True
        
        commands = [
            ("git init", "Initializing git repository"),
            ("git add .", "Adding all files to staging"),
            (f'git commit -m "Initial commit: Complete GenAI implementation"', "Creating initial commit"),
            ("git branch -M main", "Setting main branch")
        ]
        
        for command, description in commands:
            success, _ = self.run_command(command, description)
            if not success:
                print(f"❌ Failed to {description.lower()}")
                return False
        
        print("✅ Repository initialized successfully!")
        return True

    def setup_github_remote(self):
        """Set up GitHub remote repository."""
        print("\n" + "="*50)
        print("🔗 SETTING UP GITHUB REMOTE")
        print("="*50)
        
        # Check if remote already exists
        success, remotes = self.run_command("git remote -v", "Checking existing remotes")
        
        if "origin" in remotes:
            print("✅ Remote 'origin' already configured")
            return True
        
        # Add remote
        github_url = f"https://github.com/{self.github_username}/{self.repo_name}.git"
        success, _ = self.run_command(
            f"git remote add origin {github_url}",
            f"Adding GitHub remote: {github_url}"
        )
        
        if success:
            print("✅ GitHub remote configured successfully!")
            print(f"📝 Make sure you've created the repository on GitHub: {github_url}")
        
        return success

    def push_to_github(self):
        """Push repository to GitHub."""
        print("\n" + "="*50)
        print("⬆️  PUSHING TO GITHUB")
        print("="*50)
        
        # Push to GitHub
        success, _ = self.run_command(
            "git push -u origin main",
            "Pushing to GitHub (first time)"
        )
        
        if success:
            print("✅ Successfully pushed to GitHub!")
            print(f"🌐 Your repository is now available at:")
            print(f"   https://github.com/{self.github_username}/{self.repo_name}")
        else:
            print("❌ Failed to push to GitHub")
            print("💡 Make sure:")
            print("   1. You've created the repository on GitHub")
            print("   2. You have proper authentication (token/SSH)")
            print("   3. The repository name matches")
        
        return success

    def create_feature_branch(self, feature_name):
        """Create a new feature branch."""
        print(f"\n🌿 Creating feature branch: {feature_name}")
        
        # Switch to main and pull latest
        self.run_command("git checkout main", "Switching to main branch")
        self.run_command("git pull origin main", "Pulling latest changes")
        
        # Create new branch
        branch_name = f"feature/{feature_name}"
        success, _ = self.run_command(
            f"git checkout -b {branch_name}",
            f"Creating branch: {branch_name}"
        )
        
        if success:
            print(f"✅ Created and switched to branch: {branch_name}")
            print("💡 You can now make changes and commit them to this branch")
        
        return success

    def commit_changes(self, message, commit_type="feat"):
        """Commit current changes with proper message format."""
        print(f"\n💾 Committing changes: {message}")
        
        # Add all changes
        self.run_command("git add .", "Adding changes to staging")
        
        # Create commit with conventional format
        full_message = f"{commit_type}: {message}"
        success, _ = self.run_command(
            f'git commit -m "{full_message}"',
            f"Committing: {full_message}"
        )
        
        if success:
            print("✅ Changes committed successfully!")
            
            # Show current branch
            self.run_command("git branch --show-current", "Current branch")
            
            # Ask if user wants to push
            print("\n💡 Don't forget to push your changes:")
            success, branch = self.run_command("git branch --show-current", "")
            branch = branch.strip()
            print(f"   git push origin {branch}")
        
        return success

    def push_current_branch(self):
        """Push current branch to GitHub."""
        # Get current branch
        success, branch = self.run_command("git branch --show-current", "Getting current branch")
        if not success:
            return False
        
        branch = branch.strip()
        
        # Push branch
        success, _ = self.run_command(
            f"git push origin {branch}",
            f"Pushing branch '{branch}' to GitHub"
        )
        
        if success:
            print("✅ Branch pushed successfully!")
            if branch != "main":
                print(f"🔗 Create a Pull Request at:")
                print(f"   https://github.com/{self.github_username}/{self.repo_name}/compare/{branch}")
        
        return success

    def sync_with_remote(self):
        """Sync local repository with remote."""
        print("\n🔄 Syncing with remote repository")
        
        # Fetch all branches
        self.run_command("git fetch origin", "Fetching remote changes")
        
        # Switch to main and pull
        self.run_command("git checkout main", "Switching to main")
        self.run_command("git pull origin main", "Pulling latest main")
        
        print("✅ Repository synced with remote!")

    def create_release(self, version, description):
        """Create a new release."""
        print(f"\n🏷️  Creating release: {version}")
        
        # Make sure we're on main and up to date
        self.run_command("git checkout main", "Switching to main")
        self.run_command("git pull origin main", "Pulling latest changes")
        
        # Create tag
        success, _ = self.run_command(
            f'git tag -a {version} -m "{description}"',
            f"Creating tag: {version}"
        )
        
        if success:
            # Push tag
            self.run_command(
                f"git push origin {version}",
                f"Pushing tag to GitHub"
            )
            
            print("✅ Release created successfully!")
            print(f"🌐 Create GitHub release at:")
            print(f"   https://github.com/{self.github_username}/{self.repo_name}/releases/new?tag={version}")
        
        return success

    def interactive_menu(self):
        """Display interactive menu for repository management."""
        while True:
            print("\n" + "="*60)
            print("🤖 GENAI REPOSITORY MANAGEMENT")
            print("="*60)
            print("1. 📊 Check repository status")
            print("2. 🚀 Initialize repository")
            print("3. 🔗 Setup GitHub remote")
            print("4. ⬆️  Push to GitHub (first time)")
            print("5. 🌿 Create feature branch")
            print("6. 💾 Commit changes")
            print("7. 📤 Push current branch")
            print("8. 🔄 Sync with remote")
            print("9. 🏷️  Create release")
            print("10. 📖 Quick Git reference")
            print("0. ❌ Exit")
            print("="*60)
            
            choice = input("Enter your choice (0-10): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                self.check_git_status()
            elif choice == "2":
                self.initialize_repository()
            elif choice == "3":
                self.setup_github_remote()
            elif choice == "4":
                self.push_to_github()
            elif choice == "5":
                feature_name = input("Enter feature name (e.g., 'add-gpt4-vision'): ").strip()
                if feature_name:
                    self.create_feature_branch(feature_name)
            elif choice == "6":
                message = input("Enter commit message: ").strip()
                if message:
                    commit_type = input("Commit type (feat/fix/docs/chore) [feat]: ").strip() or "feat"
                    self.commit_changes(message, commit_type)
            elif choice == "7":
                self.push_current_branch()
            elif choice == "8":
                self.sync_with_remote()
            elif choice == "9":
                version = input("Enter version (e.g., 'v1.0.0'): ").strip()
                description = input("Enter release description: ").strip()
                if version and description:
                    self.create_release(version, description)
            elif choice == "10":
                self.show_git_reference()
            else:
                print("❌ Invalid choice. Please try again.")

    def show_git_reference(self):
        """Show quick Git reference."""
        reference = """
📖 QUICK GIT REFERENCE

Basic Commands:
  git status                    # Check file status
  git add .                     # Add all files
  git add file.py              # Add specific file
  git commit -m "message"      # Commit changes
  git push origin branch-name  # Push to GitHub
  git pull origin main         # Pull latest changes

Branch Management:
  git branch                   # List branches
  git checkout main            # Switch to main
  git checkout -b feature/name # Create new branch
  git merge feature/name       # Merge branch
  git branch -d feature/name   # Delete branch

Useful Commands:
  git log --oneline            # View commit history
  git diff                     # Show changes
  git stash                    # Temporarily save changes
  git stash pop                # Restore stashed changes

Emergency Commands:
  git revert commit-hash       # Undo a commit
  git reset --hard HEAD~1      # Remove last commit (careful!)
  git clean -fd                # Remove untracked files

💡 Pro Tips:
- Always pull before starting new work
- Use descriptive commit messages
- Create branches for new features
- Don't commit secrets or large files
"""
        print(reference)

def main():
    """Main function to run the GitHub manager."""
    print("🤖 GenAI Repository GitHub Management Helper")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("README.md") or not os.path.exists("requirements.txt"):
        print("❌ This doesn't appear to be the GenAI repository root directory")
        print("💡 Please run this script from the GenAi directory")
        sys.exit(1)
    
    # Initialize manager
    manager = GitHubManager()
    
    # Check if git is installed
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed or not in PATH")
        print("💡 Please install Git first: https://git-scm.com/downloads")
        sys.exit(1)
    
    # Show current status
    manager.check_git_status()
    
    # Start interactive menu
    manager.interactive_menu()

if __name__ == "__main__":
    main()
