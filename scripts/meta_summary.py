import json, glob, os, sys, collections
import h5py
root = sys.argv[1]
rows = []
for d in sorted(glob.glob(f"{root}/batch_*/*/")):
    b, ep = d.rstrip('/').split('/')[-2:]
    L = json.load(open(d+"labels.json")); S = json.load(open(d+"session.json"))
    with h5py.File(d+"camera.h5") as f:
        a = dict(f.attrs)
    rows.append(dict(batch=b, ep=ep, frames=L["frames"], excluded=L["excluded"], reviewed=L["reviewed"],
        dur=a["end_unix"]-a["start_unix"], start=a["start_unix"], cam=a["camera_name"], idx=int(a["camera_index"]),
        w=int(a["frame_width"]), dropped=int(a["dropped_frames"]), note=L.get("note",""),
        steps=[s["step"] for s in L["steps"]], missing=L.get("missing",[]),
        boards=tuple(sorted(x["board"] for x in S["devices"]))))
json.dump(rows, open(sys.argv[2],"w"), indent=0)
by = collections.defaultdict(list)
for r in rows: by[r["batch"]].append(r)
import datetime as dt
print("batch  n  excl  first(KST)  last(KST)  span_min  gap_to_next_min  med_dur_s  boards")
bs = sorted(by)
for i,b in enumerate(bs):
    R = by[b]; t0=min(r["start"] for r in R); t1=max(r["start"]+r["dur"] for r in R)
    nxt = (min(r["start"] for r in by[bs[i+1]])-t1)/60 if i+1<len(bs) else float('nan')
    durs = sorted(r["dur"] for r in R)
    f=lambda t: dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=9))).strftime("%H:%M:%S")
    print(f"{b} {len(R):3d} {sum(r['excluded'] for r in R):3d}  {f(t0)} {f(t1)} {(t1-t0)/60:6.1f} {nxt:7.1f} {durs[len(durs)//2]:6.1f}  {collections.Counter(r['boards'] for r in R).most_common(2)}")
print("cams:", collections.Counter((r["cam"],r["idx"],r["w"]) for r in rows))
print("dropped>0:", sum(r["dropped"]>0 for r in rows), " excluded:", sum(r["excluded"] for r in rows), " notes:", collections.Counter(r["note"] for r in rows).most_common(8))
print("step seqs:", collections.Counter(tuple(r["steps"]) for r in rows).most_common(5))
