#!/usr/bin/env python3
"""Current manuscript numbering without rewriting historical experiment figures."""
from pathlib import Path
import hashlib,json,re

def main():
    paper=Path('05_论文/待审/C题_Q1_英文优化稿_v3.md');out=Path('artifacts/c_q1_v3/paper_figures');out.mkdir(exist_ok=True)
    paragraphs=[p for p in paper.read_text().split('\n\n') if p.startswith('**Figure ')]
    entries=[]
    for paragraph in paragraphs:
        number=int(re.match(r'\*\*Figure (\d+)\.',paragraph).group(1))
        path=Path(re.search(r'artifacts/[^\x60]+\.pdf',paragraph).group())
        assert path.is_file(),path
        entries.append(dict(figure=number,pdf=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),caption=paragraph))
    assert [e['figure'] for e in entries]==list(range(1,7))
    (out/'inventory.json').write_text(json.dumps(dict(manuscript=str(paper),manuscript_sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),note='Authoritative current paper numbering; historical local artifact captions retain their original numbers',figures=entries),indent=2)+'\n')
    (out/'captions.md').write_text('# Current Q1 manuscript captions\n\n'+'\n\n'.join(paragraphs)+'\n')
    print('Indexed six figures in manuscript order.')

if __name__=='__main__':main()
