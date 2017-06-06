#coding: utf-8
import os

from lib_robotis import USB2Dynamixel_Device

from lib_robotis import Robotis_Servo


def find_servos(dyn):
    ''' Finds all servo IDs on the USB2Dynamixel '''
    print 'Scanning for Servos.'
    servos = []
    dyn.servo_dev.setTimeout( 0.03 ) # To make the scan faster
    for i in xrange(21):
        try:
            s = Robotis_Servo( dyn, i )
            if i>0 and i<=18:
                tipo = "body"
            if i==19 or i==20:
                tipo = "head"
#            print '\n FOUND A SERVO @ ID %d\n' % i
            servos.append( i )
        except:
            pass
    dyn.servo_dev.setTimeout( 1.0 ) # Restore to original
    return servos, tipo

def findInfo(porta):

    comando = "udevadm info -a -n "+porta

    string = os.popen(comando).read()
#    print string

    index = string.find("{serial}==")
    serial = string[index+11:index+19]
    
    index = string.find("ATTRS{idProduct}==")
    idProduct = string[index+19:index+23]

    index = string.find("ATTRS{idVendor}==")
    idVendor = string[index+18:index+22]

#    print serial, idProduct, idVendor
    return serial, idProduct, idVendor


for i in range(0,4):
    porta = "/dev/ttyUSB"+str(i)
    print porta

    try:
        dyn = USB2Dynamixel_Device(porta, 1000000)
#        print "achou"
        [servos, tipo] = find_servos( dyn )
#        print servos, tipo
        if tipo=="head":
            portahead = porta
#            print "portahead", portahead
        if tipo=="body":
            portabody = porta
#            print "portabody", portabody
    except:
        print "nao tem servo conectado"
        pass


#print "portahead", portahead
#print "portabody", portabody
try:
    HEADserial, HEADidProduct, HEADidVendor = findInfo(portahead)
    f1 = open("/etc/udev/rules.d/41-ftdi-head.rules", "w")
    f1.write("KERNEL==\"ttyUSB?\", SUBSYSTEMS==\"usb\", ATTRS{idVendor}==\""+str(HEADidVendor)+"\",  ATTRS{idProduct}==\""+str(HEADidProduct)+"\", ATTRS{serial}==\""+str(HEADserial)+"\", MODE=\"0777\", SYMLINK+=\"robot/head\"\n")
    f1.close()
except:
    print "Não foi possivel criar as regras head"
    print "Verifique se a chave que liga os servos motores está na posição ligada."
    
try:
    BODYserial, BODYidProduct, BODYidVendor = findInfo(portabody)
    f2 = open("/etc/udev/rules.d/41-ftdi-body.rules", "w")
    f2.write("KERNEL==\"ttyUSB?\", SUBSYSTEMS==\"usb\", ATTRS{idVendor}==\""+str(BODYidVendor)+"\",  ATTRS{idProduct}==\""+str(BODYidProduct)+"\", ATTRS{serial}==\""+str(BODYserial)+"\", MODE=\"0777\", SYMLINK+=\"robot/body\"")
    f2.close()
except:
    print "Não foi possivel criar as regras body"
    print "Verifique se a chave que liga os servos motores está na posição ligada."


exit(0)





