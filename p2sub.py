"""
=================================================================================================
convert a predictions file to submission format using sampleSubmission.csv
usage: python p2sub.py Submits\\sampleSubmission.csv Models\\text_body_MCMC_1.txt Submits\\Sub1_9-5-13_TextBody_MCMC.csv
=================================================================================================
"""
__author__ = 'bgregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-05-2013'

import sys
import csv
import json

print __doc__

sample_sub_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

s_f = open( sample_sub_file )
i_f = open( input_file )
o_f = open( output_file, 'wb' )

reader = csv.reader( s_f )
writer = csv.writer( o_f )

headers = reader.next()
writer.writerow( headers )

for line in reader:
	url_id = line[0]
	p = i_f.next().strip()
	writer.writerow( [ url_id, p ] )	

print "Submission file converted, saved to ",output_file
print " "
	
	