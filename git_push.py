"""
Automated Git Push Script for Vignesh-655/Human-Activity-Recognition-Sensor-Data.
"""

import os
import sys
import certifi

# Fix stdout encoding on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Fix SSL certificate verification on Windows
os.environ['SSL_CERT_FILE'] = certifi.where()

def main():
    print("=" * 60)
    print("Pushing to GitHub: Vignesh-655/Human-Activity-Recognition-Sensor-Data")
    print("=" * 60)
    
    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
            
    if not token:
        print("Error: Token is required to authenticate with GitHub.")
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
            
        print("Connecting to GitHub and pushing repository (force update)...")
        git.push(repo, auth_url, "refs/heads/main", force=True)
        print("=" * 60)
        print("SUCCESS! All project files successfully pushed to GitHub!")
        print("Repository URL: https://github.com/Vignesh-655/Human-Activity-Recognition-Sensor-Data")
        print("=" * 60)
    except Exception as e:
        print(f"Push failed: {e}")

if __name__ == "__main__":
    main()
