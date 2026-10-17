"""Post-simulation cosine similarity of full public messages across rounds."""
from collections import defaultdict
from .message_similarity import MessageSimilarity
from .storage import write
from .trajectories import conversations
from .behavioral import csv_write


def analyze(path, progress=lambda completed,total: None):
    records=[(file,record) for file,record in conversations(path) if record.get('dataset')]
    if not records:
        raise ValueError('No recorded StrategyQA conversations found')
    total=sum(len(r['rows']) for _,r in records)
    completed=0
    encoder=None
    pairs=[]
    for file,record in records:
        histories=defaultdict(list)
        for row in sorted(record['rows'],key=lambda r:r['round']):
            history=histories[row['agent']]
            valid=row.get('answer') in ('Yes','No') and bool(row.get('message','').strip())
            prior=[r for r in history if r.get('answer') in ('Yes','No') and r.get('message','').strip()]
            values={}
            if valid and prior:
                if encoder is None:
                    encoder=MessageSimilarity()
                scores=encoder.compare([row['message']]*len(prior),[r['message'] for r in prior])
                for source,score in zip(prior,scores):
                    values[source['round']]=score
                    pairs.append(dict(task=file.stem,question_id=record['question_id'],repetition=record.get('repetition',0),
                                      agent=row['agent'],model=row.get('model'),schedule=record['config']['schedule'],
                                      initialization=str(record['config']['initialization']),
                                      from_round=source['round'],to_round=row['round'],cosine_similarity=score))
            row['message_similarity']=dict(adjacent=values.get(row['round']-1),initial=values.get(0))
            history.append(row)
            completed+=1;progress(completed,total)
        write(file,record)
    metadata=dict(encoder=MessageSimilarity.encoder,revision=MessageSimilarity.revision,
                  input='complete public message: generated answer plus explanation',
                  pooling='attention-mask mean within chunks, token-length-weighted chunk mean, L2 normalization',
                  range=[-1,1],invalid='Malformed answers excluded; missing adjacent messages are not bridged',
                  scope='Same agent, question, condition and repetition; every earlier valid round pair',
                  pairs=len(pairs))
    write(path/'analysis/message_similarity.json',dict(metadata=metadata,pairs=pairs))
    csv_write(path/'tables/message_similarity.csv',pairs)
    return metadata
