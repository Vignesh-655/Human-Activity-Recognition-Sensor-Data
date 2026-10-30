"""
Helper script to push the local repository to GitHub.
Can be used with standard Git or pure-Python dulwich.
"""

import sys
import os

def push_to_github():
    print("==================================================")
    print("🚀 Git Push Helper for HAR Capstone Project")
    print("==================================================")
    
    if len(sys.argv) > 1:
        remote_url = sys.argv[1]
    else:
        remote_url = input("Enter your GitHub repository URL (e.g. https://github.com/username/har-project.git): ").strip()
        
    if not remote_url:
        print("Error: No remote URL provided.")
        return
        
    try:
        import dulwich.porcelain as git
        repo = git.Repo(".")
        
        # Check if remote exists
        config = repo.get_config()
        print(f"Setting remote 'origin' to: {remote_url}")
        config.set(("remote", "origin"), "url", remote_url.encode('utf-8'))
        config.write_to_path()
        
        print("Pushing 'main' branch to remote origin...")
        # Note: If credentials are required, dulwich will prompt or user can use git CLI / GitHub Desktop
        git.push(repo, remote_url, "refs/heads/master:refs/heads/main")
        print("✅ Successfully pushed to GitHub!")
    except Exception as e:
        print(f"\nNotice: {e}")
        print("\nAlternatively, you can push using:")
        print("1. Git CLI:")
        print(f"   git remote add origin {remote_url}")
        print("   git branch -M main")
        print("   git push -u origin main")
        print("2. Or open GitHub Desktop and click 'Publish repository'.")

if __name__ == "__main__":
    push_to_github()
