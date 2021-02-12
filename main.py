import yaml
from joblib import cpu_count
from glob import glob

print(cpu_count())
with open('config/config.yaml') as f:
    config = yaml.load(f)
train = config['train']
files_a = glob(train['files_a'])
files_b = glob(train['files_b'])
data = zip(files_a, files_b)
print(data)
print(list(data))
