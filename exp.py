import subprocess
import re

NUM_TRIALS = 100
OUTPUT_RE = re.compile(r'(?P<device>[^_]+)_sort took (?P<time>\d+.?\d*) ms')

def run(device, size):
	p = subprocess.Popen(['./sort', device, str(size)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	ret = p.wait()
	tot_time = None
	if ret != 0:
		raise RuntimeError('Invocation failed for %s:%d'%(device, size))
	for line in p.stdout.readlines():
		match = OUTPUT_RE.match(line)
		if match:
			assert match.group('device') == device
			tot_time = float(match.group('time'))
	assert(tot_time != None)
	return tot_time

def collect(device):
	avg_time = 0.0
	for i in range(NUM_TRIALS):
		avg_time += run(device, 8192)
	return (avg_time / NUM_TRIALS)

devices = ['cpu', 'gpu']
cpu_time = None
print 'Device', 'Time(ms)', 'SpeedUp'
for idevice in devices:
	avg_time = collect(idevice)
	if idevice == 'cpu':
		cpu_time = avg_time
	print idevice, '%.2f'%avg_time, '%.2f'%(cpu_time / avg_time)

