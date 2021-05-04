import json_parse
from sklearn.model_selection import train_test_split
import driver

files = ['SDSS_DATA_ALL.json']
class_lims = [10000]

if __name__ == '__main__':
    for i, file in enumerate(files):

        # x represents points, y represents labels
        tri_x, tri_y = json_parse.JSON_parse(file=file, binary=False, wise=True, class_limit = class_lims[i])
        bi_x, bi_y = json_parse.JSON_parse(file=file, binary=True, wise=True, class_limit = class_lims[i])

        # format data in dictionary form for readability
        ternary_data = dict()
        binary_data = dict()

        seed = 5

        # split data into train/test
        ternary_data['train'], ternary_data['test'], ternary_data['train_labels'], ternary_data['test_labels'] = \
            train_test_split(tri_x, tri_y, test_size=0.1, random_state=seed)

        binary_data['train'], binary_data['test'], binary_data['train_labels'], binary_data['test_labels'] = \
            train_test_split(bi_x, bi_y, test_size=0.1, random_state=seed)

        print('conducting experiment #{}'.format(i))
        driver.experiment(ternary_data, 'Ternary_Stats_{}.txt'.format(i), binary=False)
        print('Running binary experiment #{}'.format(i))
        driver.experiment(binary_data, 'Binary_Stats_{}.txt'.format(i), binary=True)
