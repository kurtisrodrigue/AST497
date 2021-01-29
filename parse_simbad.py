import glob

data_path = "data/*_Stars.txt"
txt_files = glob.glob(data_path)

print(txt_files)

for file in txt_files:
    f = open(file, "r")
    outfile = open(file.split('.')[0] + "_magnitudes.txt", "w")
    lines = f.readlines()
    for x in range(9, len(lines)-1):
        line = lines[x].split('|')
        if line[5].strip() != '~' and line[6].strip() != '~' and line[7].strip() != '~':
            outfile.write(line[5].strip() + ' ' + line[6].strip() + ' ' + line[7].strip() + '\n')

