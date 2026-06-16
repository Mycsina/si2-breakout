import subprocess, glob, yaml
h = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
for path in glob.glob("breakout_rl/configs/*.yaml"):
    c = yaml.safe_load(open(path))
    if "logic_commit" in c:
        c["logic_commit"] = h
        yaml.safe_dump(c, open(path, "w"))
print("pinned logic_commit =", h)
