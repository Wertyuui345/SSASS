import glob

files = []
for file in glob.glob("/scratch/09816/wertyuui345/ls6/AudioSeperation/training_data/mnt/g/training_data/*.npz"):
    files.append(file)

print(len(files))
