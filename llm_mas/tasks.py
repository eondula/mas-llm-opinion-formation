"""Agent-visible decision tasks, without evaluation labels or dataset evidence."""


def is_question(task):
    return task.get('kind') == 'binary_question'


def candidates(task):
    if is_question(task):
        return [choice['text'] for choice in task['choices']]
    return [task[key]['candidate'] for key in ('strategy_1', 'strategy_2')]


def question_task(qid, question, dataset='strategyqa'):
    task = dict(kind='binary_question', id=qid, dataset=dataset, question=question,
                title=question, choices=[dict(label='Yes', text='Yes'), dict(label='No', text='No')])
    validate_question(task)
    return task


def validate_question(task):
    if set(task) != {'kind', 'id', 'dataset', 'question', 'title', 'choices'}:
        raise ValueError('Question tasks accept only agent-visible fields')
    if any(not isinstance(task.get(key), str) or not task[key].strip()
           for key in ('id', 'dataset', 'question', 'title')):
        raise ValueError('Question tasks require nonempty identity and question text')
    if task['choices'] != [dict(label='Yes', text='Yes'), dict(label='No', text='No')]:
        raise ValueError('Binary question choices must map 1=Yes and 2=No')
