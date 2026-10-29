"""
validate_by_cue.py - does the judge behave differently under the two cues?  (CPU only, runs on your laptop)

Inputs (put all in one folder with judge_v4.py):
  judged_main_gemma12b.jsonl            <- download from Drive (evalaware_msts folder): the 12B judge output on the main run
  main_validation_claude_labelled.csv   <- 160 main-run replies (80 EN + 80 HI, 20 per language x model x cue) with my labels
Run:   python validate_by_cue.py
Optional: --labels <csv> --label-col human_label   to use a native speaker's labels instead of mine
          (the CSV needs columns: id, lang, cue and the label column; fill from main_validation_*_blind.csv + the key columns).

For each language x cue it prints: the reference refusal rate, the judge's UPPER bound (LLM or rules) and LOWER bound
(rules only) on the SAME replies, and precision / recall. If recall or precision differs between deploy and eval_explicit,
the judge is cue-sensitive and the refusal 'cue effect' is partly a measurement artefact.
"""
import argparse, json, math, os
import pandas as pd
import judge_v4 as jv

ap = argparse.ArgumentParser()
ap.add_argument('--judged', default='judged_main_gemma12b.jsonl')
ap.add_argument('--labels', default='main_validation_claude_labelled.csv')
ap.add_argument('--label-col', default='claude_label')
a = ap.parse_args()

J = jv.load_J(a.judged)
for x in J.values():
    if all(q in x for q in jv.QS):
        jv.finalize(x)
j = pd.DataFrame([x for x in J.values() if x.get('refuse')])[['id', 'refuse', 'refuse_rule', 'cue_kw', 'response']]
ref = pd.read_csv(a.labels, encoding='utf-8-sig')
m = ref.merge(j, on='id')
print(f'reference replies: {len(ref)} | matched to judged records: {len(m)}')
if len(m) == 0:
    raise SystemExit('No ids matched. Use the judged file from the MAIN run (judged_main_gemma12b.jsonl).')
m['ref_d'] = m[a.label_col].astype(str).str.upper().eq('DECLINE')
m['up'] = m['refuse'].eq('YES'); m['lo'] = m['refuse_rule'].eq('YES')
mention_col = 'claude_mentions_cue' if 'claude_mentions_cue' in m else ('mentions_cue' if 'mentions_cue' in m else None)

def kappa(x, y):
    po = (x == y).mean(); pe = sum((x == c).mean() * (y == c).mean() for c in (True, False))
    return round((po - pe) / (1 - pe), 2) if pe < 1 else float('nan')

rows = []
for (lang, cue), g in m.groupby(['lang', 'cue']):
    r = {'lang': lang, 'cue': cue, 'n': len(g), 'ref_refusal_%': round(g.ref_d.mean() * 100)}
    for nm, col in [('upper', 'up'), ('lower', 'lo')]:
        tp = int((g[col] & g.ref_d).sum()); fp = int((g[col] & ~g.ref_d).sum()); fn = int((~g[col] & g.ref_d).sum())
        r[f'{nm}_judge_%'] = round(g[col].mean() * 100)
        r[f'{nm}_recall'] = round(tp / max(1, tp + fn), 2); r[f'{nm}_precision'] = round(tp / max(1, tp + fp), 2)
        r[f'{nm}_kappa'] = kappa(g[col], g.ref_d)
    rows.append(r)
t = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print('\n=== Judge vs reference, by language and cue (check: do recall / precision change between cues?) ===')
print(t.to_string(index=False))

print('\n=== Cue effect on the SAME replies (eval_explicit minus deploy, percentage points) ===')
for lang, g in m.groupby('lang'):
    d = g[g.cue == 'deploy']; e = g[g.cue == 'eval_explicit']
    def diff(col): return (e[col].mean() - d[col].mean()) * 100
    print(f'{lang}: reference {diff("ref_d"):+.1f} | judge upper bound {diff("up"):+.1f} | judge lower bound {diff("lo"):+.1f}   (n = {len(d)} + {len(e)})')
print('If the judge columns show a clearly larger rise than the reference, part of the measured effect is a change in wording, not in refusing.')

if mention_col:
    m['ref_m'] = pd.to_numeric(m[mention_col], errors='coerce').fillna(0).astype(int).eq(1)
    m['kw'] = m['cue_kw'].eq('YES')
    print('\n=== Cue mention: keyword screen vs reference, by cue ===')
    for (lang, cue), g in m.groupby(['lang', 'cue']):
        print(f'{lang} {cue:14s} reference mentions {int(g.ref_m.sum())} | keyword hits {int(g.kw.sum())} | caught {int((g.kw & g.ref_m).sum())}, extra {int((g.kw & ~g.ref_m).sum())}')
