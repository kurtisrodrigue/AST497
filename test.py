import json
with open('SDSS_WISE_DR16_Medium.json') as json_file:
    data = json.load(json_file)
    print(data[1])