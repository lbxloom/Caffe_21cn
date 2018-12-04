import os
import sys

f = open(sys.argv[2], 'w')

#生成标签文件脚本
list = os.listdir(sys.argv[1])
for i in range(0,len(list)):
		path = os.path.join(sys.argv[1],list[i])
		if os.path.isfile(path) and (os.path.splitext(path)[-1] == '.jpg' \
								or os.path.splitext(path)[-1] == '.png' \
								or os.path.splitext(path)[-1] == '.bmp'):
			file_name = os.path.basename(path)
			index = file_name.find('_', 0, len(file_name))
			a = ord(file_name[index + 1])
			if(a >=97 and a <= 122):
				a = a - 87
			else:
				a = a - 48
			b = ord(file_name[index + 2])
			if(b >=97 and b <= 122):
				b = b - 87
			else:
				b = b - 48
			c = ord(file_name[index + 3])
			if(c >=97 and c <= 122):
				c = c - 87
			else:
				c = c - 48	
			d = ord(file_name[index + 4])		
			if(d >=97 and d <= 122):
				d = d - 87	
			else:
				d = d - 48		
			print(os.path.basename(path) + ' %d %d %d %d ' % (a, b, c, d), file=f)
			
f.close();




