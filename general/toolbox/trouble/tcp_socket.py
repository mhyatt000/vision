import socket
from tqdm import tqdm
import time

address = 'localhost'
port = 22  # port number is a number, not string

def connect(address,port):
    """docstring"""

    s = socket.socket()
    try:
        s.connect((address, port)) 
        print(f'{address}:{port} connection is ok')
        # originally, it was 
        # except Exception, e: 
        # but this syntax is not supported anymore. 
    except Exception as e: 
        # pass
        print("something's wrong with %s:%d. Exception is %s" % (address, port, e))
    finally:
        s.close()

def main():
    """docstring"""

    for p in tqdm(range(49000,65000)):
        connect(address,p)
        time.sleep(0.2)

if __name__ == '__main__': 
    main()
