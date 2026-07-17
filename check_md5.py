import hashlib

with open("credit_default_model.pkl", "rb") as f:
    print(hashlib.md5(f.read()).hexdigest())