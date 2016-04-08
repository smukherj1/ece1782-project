import subprocess
import re

NUM_TRIALS = 10
OUTPUT_RE = re.compile(r'(?P<device>[^\s]+) took (?P<time>\d+.?\d*) ms')

def run(device, size):

	itry = 10
	time_map = {}

	while itry >= 0:
		p = subprocess.Popen(['./sort', device, str(size)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		ret = p.wait()
		itry -= 1
		if ret == 0:
			break
		elif itry <= 0:
			raise RuntimeError('Invocation failed for %s:%d'%(device, size))
	for line in p.stdout.readlines():
		match = OUTPUT_RE.match(line)
		if match:
			case = match.group('device')
			time_ = float(match.group('time'))
			time_map[case] = time_

	assert(time_map)
	return time_map

def accum_time(time_map, case, case_time):
	if case in time_map:
		time_map[case] += case_time
	else:
		time_map[case] = case_time
	return

def report(time_map):
	base_time = time_map['cpu_sort']
	sorted_keys = sorted(time_map.keys(), key = lambda case : base_time /time_map[case])
	for icase in sorted_keys:
		print '%s\t%.6f\t%.2f'%(icase, time_map[icase],  base_time / time_map[icase])
	return

def collect(device, arrlen):
	avg_time = {}
	for i in range(NUM_TRIALS):
		case_table = run(device, arrlen)
		for icase in case_table:
			accum_time(avg_time, icase, case_table[icase])
	for icase in avg_time:
		avg_time[icase] /= NUM_TRIALS
	return avg_time



print 'Device', 'Time(ms)', 'SpeedUp'
for ilen in (16384, 100000, 1000000, 300000000):
	print '<---------Length=%d------------>'%ilen
	avg_time = collect('all', ilen)
	report(avg_time)
	

