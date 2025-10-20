import socket
import threading

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1023))
print(socket.gethostname())
s.listen(5)

clients = []
nickename = []


def Client_Handler(cli):
    pass


def BroadCasating(msg):
    pass


def recieve():
    pass


recieve()
s.close()
