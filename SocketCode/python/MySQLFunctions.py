######################################################################################
#
# MySQLFunctions.py
#
# List of functions for interacting with MySQL database
#
# Written by Christopher Neff, Matias Mendieta, Christopher Beam, and Daniel Lingerfelt
#
# For NVDA_DETEC Senior Design Project at University of North Carolina at Charlotte
#
# 19 Novemeber 2018
#
######################################################################################

maxElement = 40

import MySQLdb as mdb
from datetime import datetime

con = mdb.connect('localhost', 'cneff1', 'password', 'TestDatabase')

currentElement = 1
fulltable = 0


# Creates and empty table named UniqueID with Element, Camera, ID, Xpos, Ypos, and Time columns
def createTable():

	with con:

		cur = con.cursor()
		cur.execute("DROP TABLE IF EXISTS UniqueID")
		cur.execute("CREATE TABLE UniqueID(Element Integer, Camera Integer, ID Integer, Xpos Integer, Ypos Integer, Height Integer, Width Integer, Time TIME)")
	return

# Takes input of element and index, and fills the column at element with the data from the struct buffer at index
def addElement(element, index):

	global fulltable
	global currentElement
	if currentElement > maxElement:
		currentElement = 1
		fulltable = 1
	if fulltable == 1:

		replaceElement(index)

	if fulltable == 0:
		with con:

			camera = getCamera(index)
			id = getID(index)
			xpos = getXPos(index)
			ypos = getYPos(index)
			time = datetime.now().time()
			height = getHeight(index)
			width = getWidth(index)

			cur = con.cursor()
			cur.execute("INSERT INTO UniqueID(Element, Camera, ID, Xpos, Ypos, Height, Width, Time) VALUES( " + str(element) + ", " + str(camera) + ", " + str(id) + ", " + str(xpos) + ", " + str(ypos) + ", " + str(height) + ", " + str(width) + ", \'" + str(time) + "\')")

	currentElement += 1
	return


def fetchCamera(index):

	with con:

		cur = con.cursor()
		cur.execute("SELECT Camera FROM UniqueID WHERE Element = %s",(index, ))
		result = cur.fetchone()
		return(result[0])


def fetchID(index):

	with con:

		cur = con.cursor()
		cur.execute("SELECT ID FROM UniqueID WHERE Element = %s",(index, ))
		result = cur.fetchone()
		return(result[0])


def fetchXPos(index):

	with con:

		cur = con.cursor()
		cur.execute("SELECT XPos FROM UniqueID WHERE Element = %s",(index, ))
		result = cur.fetchone()
		return(result[0])


def fetchYPos(index):

	with con:

		cur = con.cursor()
		cur.execute("SELECT YPos FROM UniqueID WHERE Element = %s",(index, ))
		result = cur.fetchone()
		return(result[0])


def fetchHeight(index):

	with con:

		cur = con.cursor()
		cur.execute("SELECT Height FROM UniqueID WHERE Element = %s",(index, ))
		result = cur.fetchone()
		return(result[0])


def fetchWidth(index):

	with con:

		cur = con.cursor()
		cur.execute("SELECT Width FROM UniqueID WHERE Element = %s",(index, ))
		result = cur.fetchone()
		return(result[0])

# Fake replacement function
def replaceElement(index):

	print("Entered Replacement\n")
	global currentElement
	with con:


		camera = getCamera(index)
		id = getID(index)
		xpos = getXPos(index)
		ypos = getYPos(index)
		time = datetime.now().time()
		height = getHeight(index)
		width = getWidth(index)

		print("Start cursor\n")
		cur = con.cursor()
		cur.execute("UPDATE UniqueID SET Camera = %s WHERE Element = %s" % (camera,currentElement))
		cur.execute("UPDATE UniqueID SET Camera = %s WHERE Element = %s" % (id,currentElement))
		cur.execute("UPDATE UniqueID SET XPos = %s WHERE Element = %s" % (xpos,currentElement))
		cur.execute("UPDATE UniqueID SET YPos = %s WHERE Element = %s" % (ypos,currentElement))
		cur.execute("UPDATE UniqueID SET Time = \'" + str(time) + "\' WHERE Element = " + str(currentElement))
		cur.execute("UPDATE UniqueID SET Height = %s WHERE Element = %s" % (height,currentElement))
		cur.execute("UPDATE UniqueID SET Width = %s WHERE Element = %s" % (width,currentElement))

	return

# Fake function to emulate getCamera()
def getCamera(index):

	return(index*2-1)

# Fake function to emulate getID()
def getID(index):

	return(index*index + index*2+7)

# Fake function to emulate getXPos()
def getXPos(index):

	return(10*index)

# Fake function to emulate getYPos()
def getYPos(index):

	return(10*index+1)

# Fake function to emulate getHeight()
def getHeight(index):

	return(index*3)

# Fake function to emulate getWidth()
def getWidth(index):

	return(index*2+2)

# Fake Main function

createTable()

for x in range(1,200):
	addElement(x,x)

	prev = currentElement - 1
	newcamera = fetchCamera(prev)
	newid = fetchID(prev)
	newxpos = fetchXPos(prev)
	newypos = fetchYPos(prev)
	newwidth = fetchWidth(prev)
	newheight = fetchHeight(prev)

	print("For element %d, Camera = %d  ID = %d  XPos = %d  YPos = %d  Width = %d  Height = %d\n" % (prev, newcamera, newid, newxpos, newypos, newwidth, newheight))
