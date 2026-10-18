"""Primary observed state, with explicit fallback for historical saved rows."""
def measurement(row):
    return row.get('decision_time') or row.get('post',{})


def source(row):
    return 'decision_time' if row.get('decision_time') else 'post_response_probe'
