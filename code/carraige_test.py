from time import sleep


for i in range(50):
	print('test', end='\r')
	#print('Value = ' + str(i), end='\r')
	sleep(0.05)

print('End')
