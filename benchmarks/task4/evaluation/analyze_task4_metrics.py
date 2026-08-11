#!/usr/bin/env python3
"""Unified, read-only analysis of the three Task 4 result trees."""
from pathlib import Path
import argparse, ast, json, logging, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

LABELS=['higher','lower','similar']; SUBS=['rainfall','crash','activity_calendar']
ALIASES={'a':'higher','b':'lower','c':'similar','higher':'higher','increase':'higher','lower':'lower','decrease':'lower','similar':'similar','normal':'similar','same':'similar','unchanged':'similar'}
FIELDS={'qid':['question_id','id','sample_id'],'question':['question','question_text','prompt'],'options':['options','answer_options','choices'],'model':['model_name','model','model_key'],'context':['context_type','context','condition'],
 'raw':['raw_response','raw_model_output','response','model_output'],'po':['predicted_option','prediction_option'],'pl':['predicted_label','parsed_prediction_label','parsed_prediction','prediction'],
 'co':['correct_option','gold_option'],'cl':['correct_answer','ground_truth','gold_label','correct_label'],'rat':['rationale','explanation','reasoning','reasoning_text'],
 'score':['human_score','manual_score'],'old':['is_correct'],'map':['option_label_map']}

def col(d,key,default=''):
 for c in FIELDS[key]:
  if c in d:return d[c]
 return pd.Series([default]*len(d),index=d.index)
def clean(x):return '' if pd.isna(x) else str(x).strip()
def opt(x):
 m=re.fullmatch(r'(?:option\s*)?\(?\s*([abc])\s*\)?[\s\.:;,-]*',clean(x).lower());return m.group(1).upper() if m else ''
def lab(x):
 s=clean(x).lower().strip('()[]{} .,;:\"\'');return ALIASES.get(s,ALIASES.get(opt(s).lower(),''))
def omap(x):
 if isinstance(x,dict):o=x
 else:
  try:o=json.loads(clean(x))
  except Exception:
   try:o=ast.literal_eval(clean(x))
   except Exception:return {}
 return {opt(k):lab(v) for k,v in o.items() if opt(k) and lab(v)}
def read(p):
 if p.suffix.lower()=='.csv':return pd.read_csv(p)
 if p.suffix.lower()=='.jsonl':return pd.read_json(p,lines=True)
 o=json.loads(p.read_text(encoding='utf-8-sig'))
 if isinstance(o,dict):
  for k in ['data','results','predictions','records']:
   if isinstance(o.get(k),list):o=o[k];break
 return pd.DataFrame(o)
def discover(root):
 bases=[('rainfall',root/'task4_contrastive_examples'),('crash',root/'task4_contrastive_examples_2'),('activity_calendar',root/'task4_contrastive_examples_3')]; out=[]
 for s,b in bases:
  out += [(s,p) for p in b.rglob('*') if p.is_file() and p.suffix.lower() in {'.csv','.json','.jsonl'} and 'results' in [x.lower() for x in p.parts]]
 return sorted(out,key=lambda z:(SUBS.index(z[0]),str(z[1])))
def normalize(s,p,root):
 d=read(p); maps=col(d,'map').map(omap); po=col(d,'po').map(opt); co=col(d,'co').map(opt)
 pl=pd.Series([m.get(o,'') or l for m,o,l in zip(maps,po,col(d,'pl').map(lab))]); cl=pd.Series([m.get(o,'') or l for m,o,l in zip(maps,co,col(d,'cl').map(lab))])
 canonical={'A':'higher','B':'lower','C':'similar'}
 pl=pd.Series(np.where(po.ne(''),po.map(canonical),pl));cl=pd.Series(np.where(co.ne(''),co.map(canonical),cl))
 model=col(d,'model').map(clean).replace('',re.sub('_predictions$','',p.stem)); old=col(d,'old',np.nan)
 x=pd.DataFrame({'subtask':s,'context_type':col(d,'context',s).map(clean).replace('',s),'model':model,'source_file':str(p.relative_to(root)),
  'question_id':col(d,'qid').map(clean),'question':col(d,'question').map(clean),'options':col(d,'options').map(clean),'predicted_option':po,'predicted_label':pl,'correct_option':co,'correct_label':cl,
  'raw_response':col(d,'raw').map(clean),'rationale':col(d,'rat').map(clean),'human_score':col(d,'score',np.nan),'original_is_correct':old,'option_label_map':col(d,'map').map(clean)})
 x['invalid_prediction']=~x.predicted_label.isin(LABELS);x['is_correct_recomputed']=x.predicted_label.eq(x.correct_label)&~x.invalid_prediction&x.correct_label.isin(LABELS)
 x['option_map_conflict']=[bool(m) and any(m.get(k)!=v for k,v in canonical.items()) for m in maps]
 ob=old.map(lambda v:np.nan if pd.isna(v) else clean(v).lower() in {'true','1','yes'});x['is_correct_mismatch']=ob.notna()&ob.ne(x.is_correct_recomputed)
 return x
def metric(g):
 valid=~g.invalid_prediction;n=len(g);correct=int(g.is_correct_recomputed.sum()); yt=g.correct_label.where(g.correct_label.isin(LABELS),'__missing__');yp=g.predicted_label.where(valid,'__invalid__')
 p,r,f,s=precision_recall_fscore_support(yt,yp,labels=LABELS,zero_division=0)
 z={'sample_count':n,'valid_prediction_count':int(valid.sum()),'invalid_prediction_count':int((~valid).sum()),'accuracy':correct/n if n else np.nan,'strict_accuracy':correct/n if n else np.nan,'valid_only_accuracy':correct/valid.sum() if valid.sum() else np.nan,'macro_f1':f.mean()}
 for i,c in enumerate(LABELS):z.update({c+'_precision':p[i],c+'_recall':r[i],c+'_f1':f[i],c+'_support':int(s[i])})
 return z
def md(df,digits=4):
 """Small dependency-free Markdown table formatter."""
 def fmt(v):
  if pd.isna(v):return ''
  if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
  return str(v)
 headers=[str(c) for c in df.columns]; rows=[[fmt(v).replace('|','\\|').replace('\n',' ') for v in row] for row in df.itertuples(index=False,name=None)]
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(r)+' |' for r in rows])
def shared_human_sample(long,per_subtask=10,seed=5925):
 """Choose N shared IDs per subtask and expand them across every model."""
 models=sorted(long.model.unique());expected=len(models)
 q=(long.groupby(['subtask','question_id','correct_label'],observed=True)
    .agg(model_count=('model','nunique'),correct_rate=('is_correct_recomputed','mean')).reset_index())
 q=q[q.model_count.eq(expected)].copy();selected=[]
 for si,sub in enumerate(SUBS):
  pool=q[q.subtask.eq(sub)].copy();take=[]
  # Three per gold class, spanning low / medium / high empirical difficulty.
  for c in LABELS:
   g=pool[pool.correct_label.eq(c)].sort_values(['correct_rate','question_id'])
   if len(g)<3:raise ValueError(f'{sub}/{c} has fewer than three complete questions')
   pos=np.linspace(0,len(g)-1,3).round().astype(int);take.extend(g.iloc[pos].question_id.tolist())
  remain=pool[~pool.question_id.isin(take)]
  take.extend(remain.sample(n=per_subtask-len(take),random_state=seed+si).question_id.tolist())
  if len(set(take))!=per_subtask:raise AssertionError(f'{sub}: expected {per_subtask} unique IDs')
  selected.extend((sub,x) for x in take)
 chosen=pd.DataFrame(selected,columns=['subtask','question_id'])
 t=long.merge(chosen,on=['subtask','question_id'],validate='many_to_one')
 counts=t.groupby(['subtask','model'],observed=True).size()
 if len(counts)!=len(SUBS)*expected or not counts.eq(per_subtask).all():
  raise AssertionError('Each subtask/model must contain exactly the shared question count')
 return t
def make_plots(long,by,ov,dist,fd):
 fd.mkdir(exist_ok=True);plt.rcParams['figure.dpi']=140
 for y,name in [('overall_accuracy','overall_accuracy_by_model.png'),('overall_macro_f1','overall_macro_f1_by_model.png')]:
  a=ov.sort_values(y);ax=a.plot.barh(x='model',y=y,legend=False,figsize=(10,5),title=y.replace('_',' ').title());ax.set_xlim(0,1);plt.tight_layout();plt.savefig(fd/name,bbox_inches='tight');plt.close()
 for y,name in [('accuracy','accuracy_by_context_type.png'),('macro_f1','macro_f1_by_context_type.png')]:
  ax=by.pivot(index='model',columns='subtask',values=y).plot.bar(figsize=(12,6),title=y.replace('_',' ').title()+' by Context Type');ax.set_ylim(0,1);plt.xticks(rotation=30,ha='right');plt.tight_layout();plt.savefig(fd/name);plt.close()
 ax=ov.set_index('model')[[c+'_recall' for c in LABELS]].plot.bar(figsize=(12,6),title='Overall Class Recall');ax.set_ylim(0,1);plt.xticks(rotation=30,ha='right');plt.tight_layout();plt.savefig(fd/'class_recall_by_model.png');plt.close()
 cols=['predicted_'+c+'_share' for c in LABELS]+['invalid_share'];ax=dist.set_index(['model','subtask'])[cols].plot.bar(stacked=True,figsize=(15,7),title='Prediction Distribution');ax.set_ylim(0,1);plt.xticks(rotation=55,ha='right',fontsize=7);plt.tight_layout();plt.savefig(fd/'prediction_distribution_by_model_and_subtask.png');plt.close()
 for (s,m),g in long.groupby(['subtask','model']):
  cm=confusion_matrix(g.correct_label,g.predicted_label.where(~g.invalid_prediction,'invalid'),labels=LABELS+['invalid']);fig,ax=plt.subplots(figsize=(5.5,5));ax.imshow(cm,cmap='Blues');ax.set_xticks(range(4),LABELS+['invalid'],rotation=30);ax.set_yticks(range(4),LABELS+['invalid']);ax.set(xlabel='Predicted',ylabel='True',title=f'{m} — {s}')
  for i in range(4):
   for j in range(4):ax.text(j,i,cm[i,j],ha='center',va='center')
  fig.tight_layout();safe=re.sub('[^A-Za-z0-9_-]+','_',m);fig.savefig(fd/f'confusion_matrix_{safe}_{s}.png');plt.close(fig)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);a=ap.parse_args();root=a.root.resolve();out=Path(__file__).resolve().parent;logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
 frames=[];files=discover(root)
 for s,p in files:
  d=normalize(s,p,root);frames.append(d);logging.info('%s | %s | rows=%d | model=%s | fields=%s',s,p.name,len(d),','.join(d.model.unique()),'|'.join(read(p).columns))
 if not frames:raise SystemExit('No result files found')
 long=pd.concat(frames,ignore_index=True);long['subtask']=pd.Categorical(long.subtask,SUBS,ordered=True);long=long.sort_values(['subtask','model','question_id','source_file']).reset_index(drop=True);long.to_csv(out/'task4_all_predictions_long.csv',index=False,encoding='utf-8-sig')
 rows=[{'model':m,'subtask':s,**metric(g)} for (s,m),g in long.groupby(['subtask','model'],observed=True,sort=False)];by=pd.DataFrame(rows).sort_values(['subtask','model']);by.to_csv(out/'metrics_by_model_and_subtask.csv',index=False,encoding='utf-8-sig')
 ovs=[]
 for m,g in long.groupby('model'):
  z=metric(g);ovs.append({'model':m,'sample_count':z['sample_count'],'valid_prediction_count':z['valid_prediction_count'],'invalid_prediction_count':z['invalid_prediction_count'],'overall_accuracy':z['accuracy'],'overall_valid_only_accuracy':z['valid_only_accuracy'],'overall_macro_f1':z['macro_f1'],'mean_subtask_macro_f1':by[by.model.eq(m)].macro_f1.mean(),**{c+'_recall':z[c+'_recall'] for c in LABELS},**{c+'_f1':z[c+'_f1'] for c in LABELS}})
 ov=pd.DataFrame(ovs).sort_values('model');ov.to_csv(out/'overall_metrics.csv',index=False,encoding='utf-8-sig')
 cr=[]
 for (s,m),g in list(long.groupby(['subtask','model'],observed=True))+[(('overall',m),g) for m,g in long.groupby('model')]:
  z=metric(g)
  for c in LABELS:cr.append({'model':m,'subtask':s,'class':c,'precision':z[c+'_precision'],'recall':z[c+'_recall'],'f1':z[c+'_f1'],'support':z[c+'_support']})
 pd.DataFrame(cr).to_csv(out/'class_level_recall_f1.csv',index=False,encoding='utf-8-sig')
 dr=[]
 for (s,m),g in long.groupby(['subtask','model'],observed=True,sort=False):
  pred=g.predicted_label.where(~g.invalid_prediction,'invalid').value_counts();true=g.correct_label.value_counts();n=len(g);r={'model':m,'subtask':s}
  for c in LABELS:r.update({'predicted_'+c+'_count':int(pred.get(c,0)),'predicted_'+c+'_share':pred.get(c,0)/n,'true_'+c+'_share':true.get(c,0)/n,c+'_share_difference':(pred.get(c,0)-true.get(c,0))/n})
  r.update({'invalid_count':int(pred.get('invalid',0)),'invalid_share':pred.get('invalid',0)/n});dr.append(r)
 dist=pd.DataFrame(dr).sort_values(['subtask','model']);dist.to_csv(out/'prediction_distribution.csv',index=False,encoding='utf-8-sig')
 wide=by.pivot(index='model',columns='subtask',values=['accuracy','macro_f1']);wide.columns=[f'{s}_{k}' for k,s in wide.columns];wide.reset_index().merge(ov[['model','overall_accuracy','overall_macro_f1']],on='model').to_csv(out/'performance_by_context_type.csv',index=False,encoding='utf-8-sig')
 dq=[]
 for (s,m),g in long.groupby(['subtask','model'],observed=True,sort=False):dq.append({'subtask':s,'model':m,'sample_count':len(g),'duplicate_question_id_count':int(g.question_id.duplicated(keep=False).sum()),'missing_correct_answer_count':int((~g.correct_label.isin(LABELS)).sum()),'missing_prediction_count':int((g.predicted_option.eq('')&g.predicted_label.eq('')).sum()),'invalid_prediction_count':int(g.invalid_prediction.sum()),'option_map_conflict_count':int(g.option_map_conflict.sum()),'is_correct_mismatch_count':int(g.is_correct_mismatch.sum()),**{'true_'+c+'_count':int(g.correct_label.eq(c).sum()) for c in LABELS}})
 quality=pd.DataFrame(dq);quality.to_csv(out/'data_quality_report.csv',index=False,encoding='utf-8-sig')
 score_path=out/'human_reasoning_scoring_template.csv'
 if score_path.exists():
  scored=read(score_path);scored['human_score']=pd.to_numeric(scored.get('human_score'),errors='coerce');hs=scored[scored.human_score.isin([0,1,2])].copy()
 else:hs=long[pd.to_numeric(long.human_score,errors='coerce').isin([0,1,2])].copy()
 if len(hs):
  hs['human_score']=pd.to_numeric(hs.human_score);hr=[]
  groups=[('overall','overall',m,g) for m,g in hs.groupby('model')]+[('subtask',s,m,g) for (s,m),g in hs.groupby(['subtask','model'],observed=True)]
  for scope,s,m,g in groups:
   r={'scope':scope,'subtask':s,'model':m,'sample_count':len(g),'mean_score':g.human_score.mean(),'std_score':g.human_score.std(ddof=1)}
   for v in [0,1,2]:r.update({f'score_{v}_count':int(g.human_score.eq(v).sum()),f'score_{v}_share':g.human_score.eq(v).mean()})
   hr.append(r)
  human_summary=pd.DataFrame(hr).sort_values(['scope','subtask','model']);human_summary.to_csv(out/'human_reasoning_scores.csv',index=False,encoding='utf-8-sig')
  hoverall=human_summary[human_summary.scope.eq('overall')];hsub=human_summary[human_summary.scope.eq('subtask')]
  ax=hoverall.sort_values('mean_score').plot.barh(x='model',y='mean_score',xerr='std_score',legend=False,figsize=(10,5),title='Human-Evaluated Contextual Reasoning Score');ax.set_xlim(0,2);ax.set_xlabel('Mean human score (0–2)');plt.tight_layout();plt.savefig(out/'figures'/'human_reasoning_score_by_model.png',bbox_inches='tight');plt.close()
  best_h=hsub.loc[hsub.groupby('subtask').mean_score.idxmax(),['subtask','model','mean_score']]
  human_note=f"Scored rows: **{len(hs)}**. Overall and subtask summaries:\n\n{md(human_summary[['scope','subtask','model','sample_count','mean_score','std_score','score_0_count','score_0_share','score_1_count','score_1_share','score_2_count','score_2_share']])}\n\nHighest overall mean: **{hoverall.loc[hoverall.mean_score.idxmax(),'model']}** ({hoverall.mean_score.max():.4f}). Best by subtask:\n\n{md(best_h)}"
 else:
  human_note='No completed 0/1/2 human scores were found; the scoring template was generated for manual review.'
  t=shared_human_sample(long,per_subtask=10,seed=5925)[['subtask','model','question_id','question','options','correct_label','predicted_label','is_correct_recomputed','rationale','raw_response']];t['rationale']=t.rationale.mask(t.rationale.eq(''),t.raw_response);t=t.drop(columns='raw_response').rename(columns={'is_correct_recomputed':'is_correct'});t['human_score']='';t['reviewer_notes']='';t=t.sort_values(['subtask','model','question_id'])
  try:t.to_csv(out/'human_reasoning_scoring_template.csv',index=False,encoding='utf-8-sig')
  except PermissionError:
   fallback=out/'human_reasoning_scoring_template_shared10.csv';t.to_csv(fallback,index=False,encoding='utf-8-sig');logging.warning('Scoring template is open/locked; wrote %s instead',fallback)
 make_plots(long,by,ov,dist,out/'figures')
 best=by.loc[by.groupby('subtask',observed=True).accuracy.idxmax(),['subtask','model','accuracy']];lo=ov.loc[ov.similar_recall.idxmin()];inv=ov.loc[ov.invalid_prediction_count.idxmax()];bias=[]
 for c in LABELS:
  q=dist.loc[dist[c+'_share_difference'].idxmax()];bias.append(f'{c}: {q.model}/{q.subtask} ({q[c+"_share_difference"]:+.3f})')
 report=f'''# Task 4 Metrics Report\n\nPrincipal results use strict accuracy: invalid predictions count as incorrect. Labels are normalized to `higher / lower / similar`; correctness is recomputed from predicted and gold options.\n\n## 1. Overall Accuracy\n\n{md(ov[['model','overall_accuracy','overall_valid_only_accuracy','valid_prediction_count','invalid_prediction_count']])}\n\nHighest: **{ov.loc[ov.overall_accuracy.idxmax(),'model']}**.\n\n## 2. Macro-F1\n\n{md(ov[['model','overall_macro_f1','mean_subtask_macro_f1']])}\n\nHighest pooled Macro-F1: **{ov.loc[ov.overall_macro_f1.idxmax(),'model']}**.\n\n## 3. Class-Level Recall and F1\n\n{md(ov[['model','higher_recall','lower_recall','similar_recall','higher_f1','lower_f1','similar_f1']])}\n\nLowest similar recall: **{lo.model}** ({lo.similar_recall:.4f}).\n\n## 4. Performance by Context Type\n\n{md(by[['model','subtask','sample_count','valid_prediction_count','invalid_prediction_count','accuracy','macro_f1','higher_recall','lower_recall','similar_recall','higher_f1','lower_f1','similar_f1']])}\n\nBest by subtask:\n\n{md(best)}\n\n## 5. Prediction Distribution\n\nLargest positive prediction-minus-true share: {'; '.join(bias)}. Most invalid overall: **{inv.model}** ({int(inv.invalid_prediction_count)}).\n\n## 6. Human-Evaluated Contextual Reasoning Score\n\n{'Existing 0/1/2 scores were summarized.' if len(hs) else 'No human score or scored 30-item subset was found. A fixed-seed candidate set of 30 shared question IDs was expanded over all models. Raw output fills rationale where a dedicated rationale field is absent. This is a documented default, not an inferred study design.'}\n\n## Data Quality\n\nOption-map conflicts: {int(long.option_map_conflict.sum())}; original/recomputed correctness mismatches: {int(long.is_correct_mismatch.sum())}; invalid predictions: {int(long.invalid_prediction.sum())}; duplicate IDs within model/subtask: {int(quality.duplicate_question_id_count.sum())}.\n'''
 report=report.replace('No human score or scored 30-item subset was found. A fixed-seed candidate set of 30 shared question IDs was expanded over all models. Raw output fills rationale where a dedicated rationale field is absent. This is a documented default, not an inferred study design.','No human score was found. Exactly 10 shared question IDs are selected per subtask (30 distinct subtask/question pairs); all 8 models answer the same 10 questions within each subtask, producing 240 scoring rows. Raw output fills rationale where a dedicated rationale field is absent.')
 report=report.replace('Existing 0/1/2 scores were summarized.',human_note)
 (out/'task4_metrics_report.md').write_text(report,encoding='utf-8');logging.info('Models=%d; files=%d; rows=%d; outputs=%s',long.model.nunique(),len(files),len(long),out)
if __name__=='__main__':main()
