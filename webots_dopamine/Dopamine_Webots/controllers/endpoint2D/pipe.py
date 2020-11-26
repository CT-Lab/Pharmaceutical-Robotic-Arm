import os
import json
# import numpy as np


def make_pipe(pipe):
  if isinstance(pipe, str):
    try:
      os.mkfifo(pipe)
    except:
      name = pipe.split('/')[2]
      print(name, ' exist')
  elif isinstance(pipe, list):
    try:
      for p in pipe:
        os.mkfifo(p)
    except:
      print('FIFO READY')

def close_pipe(pipe):
  if isinstance(pipe, int):
    os.close(pipe)
  elif isinstance(pipe, list):
    for p in pipe:
      os.close(p)
  else:
    raise TypeError('Should be one pipe STRING or multi pipes LIST!!!', type(pipe))

def open_write_pipe(pipe):

  if isinstance(pipe, str):
    wp = os.open(pipe, os.O_SYNC | os.O_CREAT | os.O_RDWR)
    return wp
  elif isinstance(pipe, list):
    w = []
    for p in pipe:
      wp = os.open(p, os.O_SYNC | os.O_CREAT | os.O_RDWR)
      w.append(wp)
    return (w_pipe for w_pipe in w)
  else:
    raise TypeError('Open write pipe should be one pipe STRING or multi pipes LIST!!!', type(pipe))

def open_read_pipe(pipe):

  if isinstance(pipe, str):
    rp = os.open(pipe, os.O_RDONLY)
    return rp
  elif isinstance(pipe, list):
    r = []
    for p in pipe:
      rp = os.open(p, os.O_RDONLY)
      r.append(rp)
    return (r_pipe for r_pipe in r)
  else:
    raise TypeError('Open read pipe should be one pipe STRING or multi pipes LIST!!!', type(pipe))

def write_to_pipe(pipe, data):
  if isinstance(pipe, list):
    for i in range(len(pipe)):
      os.write(pipe[i], json.dumps(data[i]).encode())
  elif isinstance(pipe, int):
    os.write(pipe, json.dumps(data).encode())
  else:
    raise TypeError("Wrong Type for write pipe")

def read_from_pipe(pipe, byte=1000000):
  if isinstance(pipe, list):
    rd = []
    for i in range(len(pipe)):
      data = os.read(pipe[i], byte)
      data = json.loads(data.decode())
      rd.append(data)
    return (d for d in rd)
  elif isinstance(pipe, int):
    data = os.read(pipe, byte)
    data = json.loads(data.decode())
    return data
  else:
    raise TypeError("Wrong Type for read pipe")
