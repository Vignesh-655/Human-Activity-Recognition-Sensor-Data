"""
Automated Git Push Script for Vignesh-655/Human-Activity-Recognition-Sensor-Data.
"""

import os
import sys
import getpass
import certifi

# Fix SSL certificate verification on Windows
os.environ['SSL_CERT_FILE'] = certifi.where()

def main():
    print("=" * 60)
    print("🚀 Pushing to GitHub: Vignesh-655/Human-Activity-Recognition-Sensor-Data")
    print("=" * 60)
    
    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
    else:
        print("\nGitHub requires a Personal Access Token (PAT) for pushing.")
        print("1. Go to: https://github.com/settings/tokens/new")
        print("2. Check the 'repo' scope checkbox")
        print("3. Click 'Generate token' and paste it below:\n")
        try:
            token = input("Enter GitHub Personal Access Token (ghp_...): ").strip()
        except Exception:
            pass
            
    if not token:
        print("\n❌ Error: Token is required to authenticate with GitHub.")
        return
        
    auth_url = f"https://{token}@github.com/Vignesh-655/Human-Activity-Recognition-Sensor-Data.git"
    
    try:
        import dulwich.porcelain as git
        repo = git.Repo(".")
        
        # Ensure main branch exists
        try:
            git.branch_create(repo, 'main', force=True)
        except Exception:
            pass
            
        print("\nConnecting to GitHub and pushing files...")
        git.push(repo, auth_url, "refs/heads/main")
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! All project files successfully pushed to GitHub!")
        print("View your repository at:")
        print("👉 https://github.com/Vignesh-655/Human-Activity-Recognition-Sensor-Data")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Push failed: {e}")
        print("\nPlease check that your token has 'repo' permissions enabled.")

if __name__ == "__main__":
    main()
