#!/usr/bin/env python3

'''
AdrianHenle-L01-NameScript.py
Adrian Henle
2018.03.26
'''

from time import gmtime as gmt

# Name-printing function
def printName():
	print("Adrian Henle")
		
# Time-stamping function
def timestamp():
	t = gmt()
	print("{:04}.{:02}.{:02} {:02}:{:02}:{:02} GMT".format(
		t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
	
# __main__
if __name__ == "__main__":
	try:
		printName()
		timestamp()
	except Exception:
		try:
			print(Exception)
		except:
			print("Unspecified exception in __main__!")

			