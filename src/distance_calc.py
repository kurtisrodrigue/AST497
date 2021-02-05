from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

# calculate angular separation between two sets of coordinates
def calc_sep(ob1, ob2):
    Angle(ob1[0] + ' degrees')
    c1 = SkyCoord(Angle(ob1[0] + ' degrees'), Angle(ob1[1] + ' degrees'))
    c2 = SkyCoord(Angle(ob2[0] + ' degrees'), Angle(ob2[1] + ' degrees'))
    return c1.separation(c2)

#parse
data_file = '../data/O_Stars.txt'

# read and convert coordinates to proper form for astropy
with open(data_file, 'r') as f:
    lines = f.readlines()
    obj_list = []
    for x in range(9, len(lines)-1):
        line = lines[x].split('|')
        # if the candidate has magnitudes in B,V,R
        if line[5].strip() != '~' and line[6].strip() != '~':
            # take the coords
            coords = line[3].split()
            RA = coords[0] + ':' + coords[1] + ':' + coords[2]
            DEC = coords[3] + ':' + coords[4] + ':' + coords[5]
            obj_list.append([RA, DEC])
    for obj in obj_list:
        print(obj)
    print(len(obj_list))

neighbor_counts = []

# calculate the separation of every pair of objects
# increase a counter for every object within GBO's FOV
for obj1 in obj_list:
    counter = 0
    for obj2 in obj_list:
        sep = calc_sep(obj1, obj2)
        if sep < Angle('0:27:00 degrees'):
            counter += 1
    neighbor_counts.append((obj1, counter))
    print(counter)

#output to a text file
with open('../data/neighbor_counts.txt', 'w') as f:
    for obj in neighbor_counts:
        f.write(str(obj[0]) + ': ' + str(obj[1]) + '\n')


