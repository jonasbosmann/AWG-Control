import pyvisa, numpy as np, time

awg = pyvisa.ResourceManager().open_resource('TCPIP0::141.51.196.111::5025::SOCKET')
awg.read_termination = '\n'
awg.write_termination = '\n'
awg.timeout = 10000

PAUSE = 6  # seconds to observe scope at each step

def cmd(c):
    while True:
        e = awg.query(':SYST:ERR?')
        if e.startswith('0'): break
    awg.write(c)
    time.sleep(0.05)
    err = awg.query(':SYST:ERR?')
    print(f'  {"OK" if err.startswith("0") else "ERR"} {c.lstrip(":")} {err.strip() if not err.startswith("0") else ""}')

def upload(wave, segnum):
    data = wave.tobytes(); nb = len(data); nbs = str(nb)
    cmd(f':TRAC:DEF {segnum},2048')
    cmd(f':TRAC:SEL {segnum}')
    awg.write_raw(f':TRAC:DATA #{len(nbs)}{nbs}'.encode() + data + b'\n')
    time.sleep(0.3); awg.write('*CLS'); print('  OK    TRAC:DATA')

def make_sine(cycles=23):
    t = np.arange(2048)
    w = np.sin(2 * np.pi * cycles * t / 2048)
    return ((w + 1) * 32767.5).clip(0, 65535).astype(np.uint16)

# Step 1: setup CH2
print('\n=== Step 1: reset + setup CH2 ===')
awg.write('*CLS; *RST'); time.sleep(0.5)
cmd(':INST:CHAN 2')
cmd(':FREQ:RAST 9000000000')
cmd(':TRAC:DEL:ALL')
cmd(':INIT:CONT ON')
upload(make_sine(), segnum=2)
cmd(':FUNC:MODE:SEGM 2')
cmd(':VOLT 0.500')
cmd(':OUTP ON')
print(f'>>> CH2 should be outputting. Watching for {PAUSE}s...')
time.sleep(PAUSE)

# Step 2: switch to CH1 (no upload yet)
print('\n=== Step 2: INST:CHAN 1 + FREQ:RAST only ===')
cmd(':INST:CHAN 1')
cmd(':FREQ:RAST 9000000000')
cmd(':INIT:CONT ON')
print(f'>>> CH2 still on? Watching for {PAUSE}s...')
time.sleep(PAUSE)

# Step 3: upload waveform to CH1 segment
print('\n=== Step 3: upload segment 1 to CH1 ===')
upload(make_sine(), segnum=1)
print(f'>>> CH2 still on? Watching for {PAUSE}s...')
time.sleep(PAUSE)

# Step 4: play CH1
print('\n=== Step 4: FUNC:MODE:SEGM 1 + OUTP ON for CH1 ===')
cmd(':FUNC:MODE:SEGM 1')
cmd(':VOLT 0.500')
cmd(':OUTP ON')
print(f'>>> Both channels on? Watching for {PAUSE}s...')
time.sleep(PAUSE)

awg.close()
print('\nDiagnostic done.')