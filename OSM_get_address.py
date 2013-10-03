'''
Uses the Open Source Mapping service to get address info using long/lat coordinates from a data file,
then re-save data files with the new zip codes added.

Usage:
$ python OSM_get_address.py <input_file> <output_file> <log_file> <email_address> [delimiter]
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '10-02-2013'

import urllib2
import logging
import json
import sys
import pandas as pd

def main():
    #---Get command line params---#
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    log_file = sys.argv[3]
    email_address = sys.argv[4]
    try:
        delimiter = sys.argv[5]
    except IndexError:
        delimiter = ""
    #---Configure logging settings---#
    logging.basicConfig(filename=log_file,level=logging.DEBUG)

    #---Load input data into a dataframe---#
    if delimiter == '':
        df = pd.read_csv(input_file)
    else:
        df = pd.read_csv(input_file, delimiter)
    #---Initialize the 3 dataframe fields that we are interested in from the OSM address object---#
    df['zipcode'] = "UKNOWN"
    df['street'] = "UKNOWN"
    df['city'] = "UKNOWN"
    df['neighborhood'] = "UKNOWN"
    #---Iterate through each record in df, sending the long/lat to OSM and receiving the address in return---#
    for idx in df.index:
        ##Get OSM reverse retrieval URL
        url = "http://nominatim.openstreetmap.org/reverse?format=json&lat="+str(df['latitude'][idx])+"&lon="+str(df['longitude'][idx])\
              +"&zoom=18&addressdetails=1&email="+ email_address
        try:
            data = json.loads(urllib2.urlopen(url).read())

            df['zipcode'][idx] = data['address']['postcode']

            if "neighborhood" in data['address'].keys():
                df['neighborhood'][idx] = data['address']['neighborhood']
            elif "suburb" in data['address'].keys():
                df['neighborhood'][idx] = data['address']['suburb']

            if "road" in data['address'].keys():
                df['street'][idx] = data['address']['road']
            elif "footway" in data['address'].keys():
                df['street'][idx] = data['address']['footway']
            elif "path" in data['address'].keys():
                df['street'][idx] = data['address']['path']
            else:
                df['street'][idx] = data['address']['pedestrian']

            if "city" in data['address'].keys():
                df['city'][idx] = data['address']['city']
            elif "hamlet" in data['address'].keys():
                df['city'][idx] = data['address']['hamlet']
            else:
                df['city'][idx] = data['address']['suburb']

            msg = "Row %d, lat: %f long: %f -- Successful\n" % (idx, df['latitude'][idx],df['longitude'][idx])
            print msg
            logging.info(msg)
        except urllib2.HTTPError, e:
            error_msg = "Row %d lat: %f long: %f -- HTTP error: %d" % (idx, df['latitude'][idx],df['longitude'][idx], e.code)
            print error_msg
            logging.warning('error_msg\n')
        except urllib2.URLError, e:
            error_msg = "Row %d lat: %f long: %f -- URL error: %s" % (idx, df['latitude'][idx],df['longitude'][idx], e.reason.args[1])
            print error_msg
            logging.warning('error_msg\n')

    #---Save updated dataframe to output file---#
    if delimiter == "":
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, index=False, delimiter = delimiter)

if __name__ == "__main__":
    main()
