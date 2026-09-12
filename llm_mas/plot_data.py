"""CSV of numerical marks actually plotted (series, axis, coordinates)."""
import csv


def export(fig, path):
    rows=[]
    for index, ax in enumerate(fig.axes):
        labels={float(t):label.get_text() for t,label in zip(ax.get_xticks(),ax.get_xticklabels())}
        base=dict(axis=index,title=ax.get_title(),x_label=ax.get_xlabel(),y_label=ax.get_ylabel())
        for line in ax.lines:
            # Baseline/reference guides are not observations.
            if line.get_transform()!=ax.transData:continue
            for point,(x,y) in enumerate(zip(line.get_xdata(orig=False),line.get_ydata(orig=False))):
                rows.append(dict(base,kind='line',series=line.get_label(),point=point,x=x,y=y,x_tick_label=labels.get(float(x),'')))
        from matplotlib.collections import PathCollection
        for collection in ax.collections:
            if not isinstance(collection,PathCollection):continue
            for point,(x,y) in enumerate(collection.get_offsets()):
                colors=collection.get_array()
                rows.append(dict(base,kind='scatter',series=collection.get_label(),point=point,x=x,y=y,
                                 color_value=colors[point] if colors is not None and point<len(colors) else None))
        for image in ax.images:
            values=image.get_array()
            for y,row in enumerate(values):
                for x,value in enumerate(row):
                    rows.append(dict(base,kind='heatmap',x=x,y=y,value=value))
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(rows)
