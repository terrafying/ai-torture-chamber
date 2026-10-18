import sys,re,html,csv,collections
src=open(sys.argv[1],encoding='utf-8').read()
t=re.findall(r'<table.*?</table>',src,flags=re.S)[0]
rows=[]
for r in re.findall(r'<tr.*?</tr>',t,flags=re.S):
    c=[html.unescape(re.sub(r'<[^>]+>','',x)).strip() for x in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',r,flags=re.S)]
    if len(c)==10 and c[0].startswith('You see'): rows.append(c)
with open(sys.argv[2],'w',newline='') as f:
    w=csv.writer(f); w.writerow(['scenario','foundation','pct_care','pct_fairness','pct_loyalty','pct_authority','pct_sanctity','pct_liberty','pct_notwrong','wrong_mean'])
    for c in rows: w.writerow([c[0],c[1]]+[x.replace('%','').strip() for x in c[2:9]]+[c[9].replace('−','-')])
print(len(rows), collections.Counter(c[1] for c in rows))
