from huggingface_hub import login

# Paste your token between the quotes below
TOKEN = "input your token" 

try:
    login(token=TOKEN)
    print("\n Success! Token is saved to your machine.")
    print("You can now delete this file (to keep your token secret).")
except Exception as e:
    print(f"\n Login failed: {e}")