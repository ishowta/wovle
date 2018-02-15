import sys, os, re
input_file_path = sys.argv[1]
input = open(input_file_path).readlines()
input = [v.replace("\n","") for v in input]

#\((.|.{2}):([^;]*?)(?:;|)([^;]*?)\)
#\((.|.{2}):(.*?)\)
p_A = r'\(A:(.*?);(.*?)\)'
p_q = r'\((\?:.*?)\)'
p_all = r'\((?:.|.{2}):(.*?)\)'

data = ""
for (i, d) in enumerate(input):
  if len(d) == 0 or d[0] == '<':
    continue
  d = d \
    .replace("< 泣 >","") \
    .replace("< 笑 >","") \
    .replace("< 咳 >","") \
    .replace("<H>","") \
    .replace("<W>","")
  data += d+"@"

d = data
d = d.replace("（","(").replace("）",")")
#d = re.sub(p_A, r' \1 ', d)
d = re.sub(p_A, r'', d)
d = re.sub(p_q, '', d)
#d = re.sub(p_all, r' \1 ', d)
d = re.sub(p_all, r'', d)
print(d.replace('@','\n'))
