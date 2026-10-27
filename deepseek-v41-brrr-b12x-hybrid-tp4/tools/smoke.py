#!/usr/bin/env python3
"""Four-lane image OCR/color, JSON schema, tool round trip and real C32 checks."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import struct
import threading
import time
import urllib.request
import zlib

from config import MODELS,settings


def endpoint():
    config=settings()
    return 'http://'+os.environ.get('HOST','127.0.0.1')+':'+str(config['port'])


def request(path,payload=None,timeout=180):
    req=urllib.request.Request(endpoint()+path,None if payload is None else json.dumps(payload).encode(),
                               {'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        body=response.read()
        return json.loads(body) if body else None


def chat(content,**extra):
    payload={'model':MODELS[1],'messages':[{'role':'user','content':content}],
             'temperature':0,'max_tokens':256,
             'chat_template_kwargs':{'thinking':False,'enable_thinking':False}}
    payload.update(extra)
    return request('/v1/chat/completions',payload)


FONT={
    '0':['111','101','101','101','111'],'1':['010','110','010','010','111'],
    '2':['111','001','111','100','111'],'3':['111','001','111','001','111'],
    '4':['101','101','111','001','001'],'5':['111','100','111','001','111'],
    '6':['111','100','111','101','111'],'7':['111','001','010','010','010'],
    '8':['111','101','111','101','111'],'9':['111','101','111','001','111']}


def image_fixture(color,number):
    width=height=160
    pixels=bytearray(bytes(color)*width*height)
    foreground=(0,0,0) if color in ((0,255,0),(255,255,0)) else (255,255,255)
    for digit,char in enumerate(str(number)):
        for row,bits in enumerate(FONT[char]):
            for col,bit in enumerate(bits):
                if bit=='1':
                    for y in range(8):
                        for x in range(8):
                            at=((56+row*8+y)*width+(48+digit*32+col*8+x))*3
                            pixels[at:at+3]=bytes(foreground)
    raw=b''.join(b'\0'+pixels[y*width*3:(y+1)*width*3] for y in range(height))
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
    png+=chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(png).decode()


def run():
    assert [row['id'] for row in request('/v1/models')['data']]==MODELS
    for model in MODELS:
        answer=chat('What is 19 plus 23? Reply only with the integer.',model=model)
        assert answer['choices'][0]['message']['content'].strip()=='42'
    schema={'type':'object','properties':{'value':{'type':'integer'}},'required':['value'],'additionalProperties':False}
    reply=chat('Return JSON with value equal to 42.',response_format={'type':'json_schema',
        'json_schema':{'name':'answer','strict':True,'schema':schema}})
    assert json.loads(reply['choices'][0]['message']['content'])=={'value':42}
    tools=[{'type':'function','function':{'name':'multiply','description':'Multiply two integers',
        'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},
                      'required':['a','b'],'additionalProperties':False}}}]
    first=chat('Use the multiply tool to compute 6 times 7.',tools=tools,
               tool_choice={'type':'function','function':{'name':'multiply'}})
    assistant=first['choices'][0]['message']
    call=assistant['tool_calls'][0]
    arguments=json.loads(call['function']['arguments'])
    assert call['function']['name']=='multiply' and arguments['a']*arguments['b']==42
    follow=chat('',messages=[{'role':'user','content':'Use multiply for 6 times 7; then reply only with the result.'},
        assistant,{'role':'tool','tool_call_id':call['id'],'content':'42'}],tools=tools)
    assert follow['choices'][0]['message']['content'].strip()=='42'

    cases=[('red',(255,0,0),17),('green',(0,255,0),29),('blue',(0,0,255),42),('yellow',(255,255,0),63)]
    def vision(case):
        name,color,number=case
        content=[{'type':'text','text':'Read the large number and background color. '
                  'Reply only with JSON {"color":"...","number":0}.'},
                 {'type':'image_url','image_url':{'url':image_fixture(color,number)}}]
        response=chat(content)
        parsed=json.loads(response['choices'][0]['message']['content'])
        assert parsed=={'color':name,'number':number},parsed
        return True
    with ThreadPoolExecutor(4) as pool:assert all(pool.map(vision,cases))

    barrier=threading.Barrier(32)
    stopped=threading.Event()
    peak=[0]
    def monitor():
        while not stopped.wait(0.1):
            try:
                with urllib.request.urlopen(endpoint()+'/metrics',timeout=5) as response:lines=response.read().decode().splitlines()
                counts=[float(line.rsplit(' ',1)[1]) for line in lines
                        if not line.startswith('#') and ':num_requests_running{' in line]
                peak[0]=max([peak[0],*counts])
            except OSError:pass
    watcher=threading.Thread(target=monitor,daemon=True)
    watcher.start()
    def stream(index):
        barrier.wait(timeout=60)
        payload={'model':MODELS[1],'stream':True,'max_tokens':512,'ignore_eos':True,
            'stream_options':{'include_usage':True},'seed':178800+index,
            'messages':[{'role':'user','content':f'Write a complete Python storage-integrity test module, scenario {index}.'}],
            'chat_template_kwargs':{'thinking':False,'enable_thinking':False}}
        req=urllib.request.Request(endpoint()+'/v1/chat/completions',json.dumps(payload).encode(),
                                   {'Content-Type':'application/json'})
        pieces=[]
        with urllib.request.urlopen(req,timeout=300) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
                data=json.loads(line[6:])
                for choice in data.get('choices',[]):pieces.append(choice.get('delta',{}).get('content') or '')
        assert len(''.join(pieces))>32
        return True
    try:
        with ThreadPoolExecutor(32) as pool:assert all(pool.map(stream,range(32)))
    finally:stopped.set();watcher.join(timeout=10)
    assert peak[0]==32,f'Observed only {peak[0]} active requests; C32 was not verified'
    def arithmetic(index):
        value=101+index
        response=chat(f'Compute {value} times 13 plus 7. Reply only with the integer.')
        assert response['choices'][0]['message']['content'].strip()==str(value*13+7)
        return True
    with ThreadPoolExecutor(32) as pool:assert all(pool.map(arithmetic,range(32)))
    return {'passed':True,'aliases':2,'json_schema':True,'tool_round_trip':True,
            'vision_ocr_color':'4/4','observed_active_requests':peak[0],'arithmetic':'32/32'}


if __name__=='__main__':print(json.dumps(run(),indent=2))
