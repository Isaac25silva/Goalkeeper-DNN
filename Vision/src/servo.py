import ctypes

"""Describes Class Servo - Communicate, Write and read from Vision Servo-motors"""
class Servo(object):
	"""Initializes the communication with the servos"""
	def __init__(self,posTILT,posPAN):
		"""print Start the Class Servo"""
		print "Start the Class Servo"
		""" Using shared memory from C++ functions"""
		""" Loads Vision library in the path /build/lib/libvision.so (Needs compilation and build) """
		ctypes.cdll.LoadLibrary("../../build/lib/libvision.so")
		""" Calls lybrary that contains C++ functions """
		self.visionlib = ctypes.CDLL('../../build/lib/libvision.so') 
		""" Defines return type (boolean)  """
		self.visionlib.initServo.restype = ctypes.c_bool 
		"""Conects to Pan and tilt servos"""
		if self.visionlib.initServo(ctypes.c_int(posTILT), ctypes.c_int(posPAN)):
			""" Error treatament, if not connected to servos exit(0)"""  
			exit (0)
		""" Define return type of the method dxlReadByte (int)"""
		self.visionlib.dxlReadByte.restype = ctypes.c_int 
		""" Define return type of the method dxlReadWord (int)"""
		self.visionlib.dxlReadWord.restype = ctypes.c_int 
		""" Define initial torque with 50% to servo ID=20, parameter 34 seted with 512"""
		self.visionlib.dxlWriteWord(ctypes.c_int(20), ctypes.c_int(34), ctypes.c_int(512)); 
		""" Define initial torque with 50% to servo ID=19, parameter 34 seted with 512"""
		self.visionlib.dxlWriteWord(ctypes.c_int(19), ctypes.c_int(34), ctypes.c_int(512)); 
		#--------------------------------------------------------------------------------------------------------------------
		""" Reads a byte from servo defined by ID """
	def readByte(self, ID, Pos):
		"""Returns a byte from servo defined by ID in pos position"""
		return self.visionlib.dxlReadByte( ctypes.c_int(ID), ctypes.c_int(Pos))
		""" Reads a word from servo defined by ID """
	def readWord(self, ID, Pos):
		""" Returns a word from servo defined by ID in pos position"""
		return self.visionlib.dxlReadWord( ctypes.c_int(ID), ctypes.c_int(Pos))
		""" Writes a byte from servo defined by ID """
	def writeByte(self, ID, Pos, value):
		""" Writes a byte in servo ID, Position Pos, and the value to be written"""
		self.visionlib.dxlWriteByte( ctypes.c_int(ID), ctypes.c_int(Pos), ctypes.c_int(value))
		""" Writes a word from servo defined by ID """
	def writeWord(self, ID, Pos, value):
		""" Writes a word in servo ID, Position Pos, and the value to be written"""
		self.visionlib.dxlWriteWord( ctypes.c_int(ID), ctypes.c_int(Pos), ctypes.c_int(value))
