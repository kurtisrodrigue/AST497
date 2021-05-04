import json

def check_point(point):
    for i in range(len(point)):
        if point[i] > 50:
            return True
    return False


def JSON_parse(file, binary = False, wise=False, class_limit=1000):
    x = []
    y = []
    # count galaxy, star, qso
    counts = {'QSO' : 0, 'GALAXY': 0, 'STAR': 0, 'NOT QSO':0}
    with open(file) as json_file:
        data = json.load(json_file)
        length = len(data[0]['Rows'])
        for i, row in enumerate(data[0]['Rows']):
            if counts[row['class']] < class_limit:
                if wise:
                    temp_arr = [row['u'],row['g'],row['r'],
                                row['i'],row['z'],row['w1'],
                                row['w2'],row['w3'],row['w4'],
                                row['psfMag_r'], row['psfMag_z']]
                else:
                    temp_arr = [row['u'], row['g'], row['r'],
                                row['i'], row['z']]
                if not check_point(temp_arr):
                    if binary:
                        if row['class'] == "QSO":
                            y.append(row['class'])
                            counts[row['class']] += 1
                            x.append(temp_arr)
                        else:
                            if counts['NOT QSO'] < class_limit:
                                y.append("NOT QSO")
                                counts['NOT QSO'] += 1
                                x.append(temp_arr)
                    else:
                        x.append(temp_arr)
                        y.append(row['class'])
                        counts[row['class']] += 1
    print('Counts: QSO: {}, GAL: {}, STAR: {}, NOT QSO: {}'.format(counts['QSO'], counts['GALAXY'],
                                                                   counts['STAR'], counts['NOT QSO']))
    return x, y

