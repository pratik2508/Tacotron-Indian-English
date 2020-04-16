import argparse
import falcon
from hparams import hparams, hparams_debug_string
import os
import numpy as np
from synthesizer import Synthesizer
from util import audio
import io
from flask import Flask, render_template, request, make_response


app = Flask(__name__, template_folder='template')
synthesizer = Synthesizer()




@app.route('/synthesize', methods=['GET'])
def synth():
  # response = make_response((synthesizer.synthesize(request.args.get("text"))).getvalue())
  splits = str(request.args.get("text")).replace(',', '.').replace('?', '.').replace('!', '.').replace(';', ',')
  print(len(splits))
  if len(splits) < 15:
    splits = str(request.args.get("text")).replace(',', '.').replace('?', '.').replace('!', '.').replace(':', ',').replace(';', ',')
    splits = splits + str(', ')
    print(splits)
    result = io.BytesIO()
    res = synthesizer.synthesize1(splits)
    print(res.shape)
    audio.save_wav(res, result)
    result_final = make_response((result.getvalue()))



  else:
    splits = str(request.args.get("text")).replace(', ', '. ').replace('-', '.').replace('?', '.').replace('!', '.').\
      replace(';', ',').replace('/', ',').split('. ')

    print(splits)

    result=io.BytesIO()

    res1 = np.empty(shape=(0, 2))
    j = ''
    for iter, i in enumerate(splits[0:]):
      # if iter == 1:
      #   i = j + str(', ') + i

      # i = j + str(',') + i
      # j= ''

      if j != '':
        i = j + str(', ') + i
        j = ''
      else:
        pass

      if len(i) < 15:
        a = iter
        j = i
        continue

      print(i, len(i))



      # response = make_response((synthesizer.synthesize(i).getvalue()))
      res = synthesizer.synthesize1(i)
      print(res.shape)
      res1 = np.append(res1, res)  # res,res1 both are numpy array objects
    audio.save_wav(res1, result)  # here res1 is numpy obj result is bytes.io obj

    result_final = make_response((result.getvalue()))   # result_final is response obj

  result_final.headers['Content-Type'] = 'audio/wav'
  return result_final


@app.route('/')
def UIRe():
  # return  UIRes
  return render_template('jn.html')


if __name__ == '__main__':

  from wsgiref import simple_server
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
  parser.add_argument('--port', type=int, default=9000)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')

  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  # os.environ["CUDA_VISIBLE_DEVICES"]=""     # loading model in cpu
  hparams.parse(args.hparams)
  print(hparams_debug_string())

  synthesizer.load(args.checkpoint)
  print('Serving on port %d' % args.port)
  app.run(host='0.0.0.0',port=5000)
else:
  synthesizer.load(os.environ['CHECKPOINT'])
