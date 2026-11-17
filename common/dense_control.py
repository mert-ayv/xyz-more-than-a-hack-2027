"""Dense all-player ownership and directed-transition model; external development data only.

Build caches / score with run.sh's pinned environment. Train / predict with the existing
numpy+torch environment (or CUDA). No learned upstream inputs, target labels or reserved games.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
TRAIN_SEEDS = (100, 101, 102, 103)
EVAL_SEEDS = (0, 1, 2)
GAMES = ("metrica_1", "metrica_2", "dfl_J03WMX", "dfl_J03WN1", "dfl_J03WOH", "dfl_J03WOY", "dfl_J03WPY")
G_COLUMNS = ("seen", "since", "until", "bx", "by", "bvx", "bvy", "speed", "accel", "turn_cos",
             "nearest_h", "nearest_a", "count_h", "count_a", "inside", "restart_h", "restart_a",
             "restart_unknown", "signed_nearest_restart", "future_displacement")
P_COLUMNS = ("px", "py", "rx", "ry", "distance", "vx", "vy", "closing", "path_m10", "path_m5",
             "path_p5", "path_p10", "intercept_p5", "intercept_p10", "min_path", "velocity_cos",
             "ball_seen", "velocity_valid")
CONFIG = dict(version=1, width=96, player_width=32, blocks=7, window=1024, batch=32, steps=1500,
              lr=.002, transition_weight=.5, positive_weight=50., boundary_radius=2,
              halo=127, train_seeds=TRAIN_SEEDS, eval_seeds=EVAL_SEEDS)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n"); tmp.replace(path)


def save_npy(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f: np.save(f, value)
    tmp.replace(path)


def source_identity():
    paths = [ROOT/"common/tuning_data.py", ROOT/"common/dfl_data.py",
             ROOT/"evaluation/possession_metrics.py"]
    h = hashlib.sha256()
    h.update(inspect.getsource(features).encode())
    for p in paths: h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def ensure_plan(out):
    path = out/"plan.json"
    if path.exists():
        plan=json.loads(path.read_text())
        if (ROOT/"common/tuning_data.py").exists():
            assert plan["source_identity"]==source_identity(), "feature/loader/scorer identity changed"
        return plan
    plan = dict(config=CONFIG, source_identity=source_identity(), build_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), games=GAMES,
        inputs="Only external degraded tracking and supplied restarts; original unsmoothed frame labels.",
        features=dict(G=G_COLUMNS,P=P_COLUMNS,mask="All finite active players; dynamic per-item player capacity=max active count across teams, batches pad with masked zeros. No truncation."),
        split="Seven outer LOGO folds. Geometry has no trained stage. Normalisation, weights and decoder calibration exclude the outer held-out game. Each outer fold also trains one five-game inner model, holding out the next cyclic game for calibration on seed104.",
        schedule="1500 AdamW/OneCycle steps,32x1024 dense frames/step, same frame budget as historical frame net; fixed seed7, random pitch/team symmetries.",
        sampling="Per game/seed balanced. Half uniform windows, quarter true short-spell centres, quarter true transitions. No original label changes.",
        targets="Owner H/A/D at every frame, plus six directed transitions y[t-1]→y[t] with ±2-frame positives; independent direction channels, periods/X masked.",
        ablations="Same joint model: constant-cost owner Viterbi vs directed-head Viterbi. Separates inference role of transition head, not auxiliary training loss.",
        decoder="Inner-held-out seed104 calibrates each direction with regularised Platt scaling. Baseline HA cost grid1/2/4/8/16, LD2/4/8/16/32. Direction weights0/0.25/0.5/1 chosen on same inner game at matched baseline costs; zero is a no-head fallback. Direction contributes calibrated loglikelihood ratio against inner prevalence, clipped±8. Dense25Hz; no peak selection/minimum duration.",
        validation="Original scorer, eval seeds0–2 on seven reused development games. Report spellF1/per-game/short precision+recall. Compare frozen historical champion separately; its upstream stack is non-nested.",
        untouched="Public BAR-ATH labels and target match not loaded; two DFL reserved games excluded. Previous target inspection remains development exposure.",
        paid_compute="Only parent-managed compute within existing50USD total project cap.")
    dump(path,plan); return plan


def features(match):
    """Pure geometry at original25Hz, preserving every finite active player and masks."""
    n,fps=match.n_frames,match.fps
    scale=np.asarray(match.pitch,np.float32)
    raw=match.ball*scale-scale/2
    seen=np.isfinite(raw).all(1)
    ball=np.zeros_like(raw); since=np.full(n,250,np.float32); until=since.copy()
    for a,b in match.periods:
        ix=np.flatnonzero(seen[a:b])+a
        if not len(ix): continue
        for j in range(2): ball[a:b,j]=np.interp(np.arange(a,b),ix,raw[ix,j])
        prev=np.maximum.accumulate(np.where(seen[a:b],np.arange(a,b),a-250))
        nxt=np.minimum.accumulate(np.where(seen[a:b],np.arange(a,b),b+250)[::-1])[::-1]
        since[a:b]=np.minimum(np.arange(a,b)-prev,250)
        until[a:b]=np.minimum(nxt-np.arange(a,b),250)
    bv=np.zeros_like(ball); acc=np.zeros(n,np.float32)
    path=[]
    for dt in (-10,-5,5,10):
        shifted=ball.copy()
        for a,b in match.periods: shifted[a:b]=ball[np.clip(np.arange(a,b)+dt,a,b-1)]
        path.append(shifted)
    for a,b in match.periods:
        if b-a>2:
            bv[a:b]=np.gradient(ball[a:b],axis=0)*fps
            acc[a:b]=np.linalg.norm(np.gradient(bv[a:b],axis=0),axis=1)*fps
    bv=np.clip(bv,-60,60)
    speed=np.linalg.norm(bv,axis=1)
    before=bv.copy(); after=bv.copy()
    for a,b in match.periods:
        before[a:b]=bv[np.clip(np.arange(a,b)-5,a,b-1)]
        after[a:b]=bv[np.clip(np.arange(a,b)+5,a,b-1)]
    turn=(before*after).sum(1)/(np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1)+1e-3)
    pos=match.players*scale-scale/2
    valid=np.isfinite(pos).all(2)
    stable=valid.copy(); stable[1:] &= valid[:-1]; stable[:-1] &= valid[1:]
    for a,b in match.periods: stable[a]=False; stable[b-1]=False
    pv=np.zeros_like(pos)
    for a,b in match.periods:
        if b-a>2: pv[a:b]=np.nan_to_num(np.gradient(pos[a:b],axis=0))*fps
    pv=np.where(stable[...,None],np.clip(pv,-40,40),0)
    rel=pos-ball[:,None,:]; dist=np.linalg.norm(rel,axis=2)
    closing=-(rel*(pv-bv[:,None,:])).sum(2)/np.maximum(dist,1e-3)
    ds=[np.linalg.norm(pos-p[:,None,:],axis=2) for p in path]
    intercept=[np.linalg.norm(pos+pv*(dt/fps)-path[j][:,None,:],axis=2) for dt,j in ((5,2),(10,3))]
    velocity_cos=(pv*bv[:,None,:]).sum(2)/(np.linalg.norm(pv,axis=2)*speed[:,None]+1e-3)
    pc=np.stack([pos[:,:,0]/52.5,pos[:,:,1]/34,rel[:,:,0]/10,rel[:,:,1]/10,np.minimum(dist,60)/10,
                 pv[:,:,0]/10,pv[:,:,1]/10,np.clip(closing,-60,60)/10]
                +[np.minimum(d,60)/10 for d in ds+intercept]
                +[np.minimum(np.minimum.reduce(ds),60)/10,velocity_cos,
                  np.broadcast_to(seen[:,None],valid.shape),stable],axis=2)
    capacity=max(int(valid[:,match.team==t].sum(1).max()) for t in (0,1))
    P=np.zeros((n,2,capacity,len(P_COLUMNS)),np.float16); M=np.zeros((n,2,capacity),bool)
    nearest=[]; counts=[]; maximum=[]
    for team in (0,1):
        ids=np.flatnonzero(match.team==team); vm=valid[:,ids]
        count=vm.sum(1); maximum.append(int(count.max()))
        assert count.max()<=capacity, "packing would truncate"
        order=np.argsort(~vm,axis=1,kind="stable")[:,:capacity]
        if order.shape[1]<capacity: order=np.pad(order,((0,0),(0,capacity-order.shape[1])),mode="edge")
        rows=np.arange(n)[:,None]
        P[:,team]=np.nan_to_num(pc[:,ids][rows,order]).astype(np.float16)
        M[:,team]=np.arange(capacity)[None,:]<count[:,None]
        P[:,team]*=M[:,team,:,None]
        nearest.append(np.minimum(np.where(vm,dist[:,ids],np.inf).min(1),60)/30)
        counts.append(count/11)
    restart=np.zeros((n,3),np.float32); signed=np.ones(n,np.float32)
    if match.restarts:
        frames=np.array([int(r["frame"]) for r in match.restarts])
        teams=np.array([int(r["team"]) if int(r["team"]) in (0,1) else 2 for r in match.restarts])
        for a,b in match.periods:
            rr=(frames>=a)&(frames<b)
            if not rr.any(): continue
            local=frames[rr]; tm=teams[rr]
            delta=np.arange(a,b)[:,None]-local[None,:]
            near=np.abs(delta).argmin(1); signed[a:b]=np.clip(delta[np.arange(b-a),near]/250,-1,1)
            for t in range(3):
                if (tm==t).any(): restart[a:b,t]=np.maximum(0,1-np.abs(delta[:,tm==t]).min(1)/50)
    G=np.column_stack([seen,since/250,until/250,ball[:,0]/52.5,ball[:,1]/34,bv[:,0]/20,bv[:,1]/20,
        speed/20,np.minimum(acc,500)/100,turn,*nearest,*counts,
        ((np.abs(ball[:,0])<=scale[0]/2)&(np.abs(ball[:,1])<=scale[1]/2)),restart,signed,
        np.linalg.norm(path[-1]-ball,axis=1)/10]).astype(np.float16)
    assert G.shape==(n,20) and np.isfinite(G).all() and np.isfinite(P).all()
    return G,P,M,dict(max_active_per_team=maximum,player_capacity=capacity,input_track_slots=len(match.team),fps=fps)


def build(args):
    sys.path.insert(0,str(ROOT))
    from common.compare import load_games
    from common.tuning_data import degrade
    plan=ensure_plan(args.out); tick=time.monotonic()
    for game,(clean,gt) in load_games("all",holdout=False).items():
        if args.games and game not in args.games: continue
        for seed in args.seeds:
            path=args.out/"cache"/f"{game}-{seed}"; done=path/"meta.json"
            if done.exists(): print("cached",path.name,flush=True); continue
            path.mkdir(parents=True,exist_ok=True)
            G,P,M,meta=features(degrade(clean,seed))
            for name,value in dict(G=G,P=P,M=M,gt=gt.astype(np.uint8),periods=np.array(clean.periods)).items():
                save_npy(path/f"{name}.npy",value)
            meta.update(game=game,seed=seed,source_identity=plan["source_identity"],
                original_gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),
                bytes=sum(p.stat().st_size for p in path.glob("*.npy")))
            dump(done,meta)
            print(json.dumps(dict(item=path.name,seconds=time.monotonic()-tick,**meta)),flush=True)


def read_item(path):
    meta=json.loads((path/"meta.json").read_text())
    return dict(path=str(path),meta=meta,**{name:np.load(path/f"{name}.npy",mmap_mode="r")
                for name in ("G","P","M","gt","periods")})


def directed_targets(gt,periods,radius=2):
    target=np.zeros((len(gt),6),np.uint8)
    for a,b in periods:
        changed=np.flatnonzero((gt[a+1:b]!=gt[a:b-1])&(gt[a+1:b]<3)&(gt[a:b-1]<3))+a+1
        for t in changed:
            k=PAIRS.index((int(gt[t-1]),int(gt[t])))
            target[max(a,t-radius):min(b,t+radius+1),k]=1
    target[gt>=3]=0
    return target


def transition_valid(gt,periods):
    valid=np.zeros(len(gt),bool)
    for a,b in periods: valid[a+1:b]=(gt[a+1:b]<3)&(gt[a:b-1]<3)
    return valid


def make_net(torch,config):
    nn=torch.nn
    class Dense(nn.Module):
        def __init__(self):
            super().__init__(); pw=config["player_width"]; width=config["width"]
            self.register_buffer("g_mu",torch.zeros(20)); self.register_buffer("g_sd",torch.ones(20))
            self.register_buffer("p_mu",torch.zeros(18)); self.register_buffer("p_sd",torch.ones(18))
            self.player=nn.Sequential(nn.Linear(18,pw),nn.ReLU(),nn.Linear(pw,pw),nn.ReLU())
            self.attention=nn.Linear(pw,1)
            self.input=nn.Conv1d(20+6*pw,width,1)
            self.blocks=nn.ModuleList([nn.Conv1d(width,width,3,padding=2**i,dilation=2**i) for i in range(config["blocks"])])
            self.owner=nn.Conv1d(width,3,1); self.transition=nn.Conv1d(width,6,1)
        def forward(self,G,P,M):
            h=self.player((P-self.p_mu)/self.p_sd)
            mask=M[...,None]
            mean=(h*mask).sum(3)/mask.sum(3).clamp(min=1)
            mx=h.masked_fill(~mask,-1e4).max(3).values
            mx=torch.where(M.any(3)[...,None],mx,0)
            attention=torch.softmax(self.attention(h).squeeze(-1).masked_fill(~M,-1e4),dim=3)
            attention=attention*M; attention=attention/attention.sum(3,keepdim=True).clamp(min=1e-6)
            weighted=(h*attention[...,None]).sum(3)
            x=torch.cat([(G-self.g_mu)/self.g_sd,mean.flatten(2),mx.flatten(2),weighted.flatten(2)],dim=2)
            x=self.input(x.transpose(1,2))
            for block in self.blocks: x=x+torch.relu(block(x))
            x=torch.relu(x)
            return self.owner(x).transpose(1,2),self.transition(x).transpose(1,2)
    return Dense()


def normalisation(items):
    g=np.concatenate([np.asarray(it["G"][::25],np.float32) for it in items])
    p=np.concatenate([np.asarray(it["P"][::25],np.float32)[it["M"][::25]] for it in items])
    return dict(g_mu=g.mean(0),g_sd=g.std(0)+.01,p_mu=p.mean(0),p_sd=p.std(0)+.01)


def prepare(items):
    for it in items:
        gt=it["gt"]; periods=it["periods"].tolist(); centres=[]; transitions=[]
        for a,b in periods:
            changes=np.flatnonzero(gt[a+1:b]!=gt[a:b-1])+a+1
            edges=np.r_[a,changes,b]
            centres.extend([(int(s+e)//2,a,b) for s,e in zip(edges[:-1],edges[1:]) if e-s<125 and gt[s]<2])
            transitions.extend([(int(t),a,b) for t in changes if gt[t]<3 and gt[t-1]<3])
        it.update(target=directed_targets(gt,periods),tvalid=transition_valid(gt,periods),short=centres,changes=transitions)


def batch(items,rng,config):
    G,P,M,Y,T,V=[],[],[],[],[],[]; win=config["window"]
    for _ in range(config["batch"]):
        it=items[int(rng.integers(len(items)))]; mode=rng.random()
        points=it["short"] if mode<.25 else it["changes"] if mode<.5 else []
        if points:
            centre,a,b=points[int(rng.integers(len(points)))]; s=int(np.clip(centre-win//2,a,b-win))
        else:
            periods=[(int(a),int(b)) for a,b in it["periods"] if b-a>=win]
            a,b=periods[int(rng.integers(len(periods)))]; s=int(rng.integers(a,b-win+1))
        G.append(np.array(it["G"][s:s+win],np.float32)); P.append(np.array(it["P"][s:s+win],np.float32))
        M.append(np.array(it["M"][s:s+win])); Y.append(np.array(it["gt"][s:s+win],np.int64))
        T.append(it["target"][s:s+win].astype(np.float32))
        V.append(it["tvalid"][s:s+win])
    capacity=max(p.shape[2] for p in P)
    P=[np.pad(p,((0,0),(0,0),(0,capacity-p.shape[2]),(0,0))) for p in P]
    M=[np.pad(m,((0,0),(0,0),(0,capacity-m.shape[2]))) for m in M]
    G,P,M,Y,T,V=map(np.stack,(G,P,M,Y,T,V))
    xf,yf=bool(rng.integers(2)),bool(rng.integers(2))
    if xf:
        G=G[:,:,[0,1,2,3,4,5,6,7,8,9,11,10,13,12,14,16,15,17,18,19]]
        G[:,:,[3,5]]*=-1; P=P[:,:,::-1].copy(); M=M[:,:,::-1].copy(); P[:,:,:,:,[0,2,5]]*=-1
        Y=np.array([1,0,2,3])[Y]
        order=[PAIRS.index((1-a if a<2 else a,1-b if b<2 else b)) for a,b in PAIRS]
        T=T[:,:,order]
    if yf: G[:,:,[4,6]]*=-1; P[:,:,:,:,[1,3,6]]*=-1
    return G,P,M,Y,T,V


def torch_setup(threads,seed=7):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.set_num_threads(threads); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False
    return torch


def atomic_torch_save(torch,path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp"); torch.save(value,tmp); tmp.replace(path)


def train(args):
    plan=ensure_plan(args.out); config=dict(plan["config"])
    config.update(batch=args.batch or config["batch"],steps=args.steps or config["steps"])
    assert args.holdout in GAMES
    inner=GAMES[(GAMES.index(args.holdout)+1)%len(GAMES)]
    excluded={args.holdout,inner} if args.role=="inner" else {args.holdout}
    folder=args.out/"folds"/args.holdout/args.role; checkpoint=folder/"checkpoint.pt"
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch=torch_setup(args.threads); device=torch.device(args.device)
    if (folder/"model.pt").exists():
        done=torch.load(folder/"model.pt",map_location="cpu",weights_only=False)
        assert done["config"]==config and done["source_identity"]==plan["source_identity"]
        assert done["training_code_sha256"]==code_sha256 and done["holdout"]==args.holdout and done["role"]==args.role
        assert done["train_games"]==[g for g in GAMES if g not in excluded] and tuple(done["train_seeds"])==TRAIN_SEEDS
        print(json.dumps(dict(event="completed_stage_skip",holdout=args.holdout,role=args.role)),flush=True)
        return
    paths=[args.out/"cache"/f"{g}-{s}" for g in GAMES if g not in excluded for s in TRAIN_SEEDS]
    items=[read_item(p) for p in paths]
    assert all(it["meta"]["game"]!=args.holdout for it in items)
    assert all(it["meta"]["source_identity"]==plan["source_identity"] for it in items)
    prepare(items)
    model=make_net(torch,config).to(device)
    for key,value in normalisation(items).items(): getattr(model,key).copy_(torch.from_numpy(value).to(device))
    opt=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=.001)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,config["lr"],total_steps=config["steps"])
    rng=np.random.default_rng(7); start=0
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        assert ck["config"]==config and ck["source_identity"]==plan["source_identity"]
        assert ck["training_code_sha256"]==code_sha256, "training code changed since checkpoint"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start=ck["step"]; rng.bit_generator.state=ck["numpy_rng"]; torch.set_rng_state(ck["torch_rng"].cpu())
        if device.type=="cuda" and ck.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(ck["cuda_rng"])
    halo=config["halo"]; tick=time.monotonic()
    progress=folder/"progress.json"
    history=json.loads(progress.read_text()).get("history",[]) if start and progress.exists() else []
    history=[row for row in history if row["step"]<=start]
    previous_elapsed=history[-1]["elapsed"] if history else 0
    print(json.dumps(dict(event="start",holdout=args.holdout,device=str(device),torch=torch.__version__,config=config,
        parameters=sum(p.numel() for p in model.parameters()),train_games=[g for g in GAMES if g not in excluded],
        role=args.role,inner_holdout=inner,training_code_sha256=code_sha256,
        frames=sum(sum(int(b-a) for a,b in it["periods"]) for it in items),resume_step=start)),flush=True)
    for step in range(start,config["steps"]):
        arrays=batch(items,rng,config)
        G,P,M,Y,T,V=[torch.from_numpy(v).to(device) for v in arrays]
        owner,transition=model(G,P,M)
        owner,transition,Y,T,V=owner[:,halo:-halo],transition[:,halo:-halo],Y[:,halo:-halo],T[:,halo:-halo],V[:,halo:-halo]
        valid=Y<3; targets=Y.masked_fill(~valid,-100)
        owner_loss=torch.nn.functional.cross_entropy(owner.reshape(-1,3),targets.reshape(-1),ignore_index=-100)
        transition_loss=torch.nn.functional.binary_cross_entropy_with_logits(transition,T,
            pos_weight=torch.tensor(config["positive_weight"],device=device),reduction="none")
        transition_loss=(transition_loss*V[...,None]).sum()/(V.sum()*6).clamp(min=1)
        loss=owner_loss+config["transition_weight"]*transition_loss
        assert torch.isfinite(loss), "nonfinite loss"
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step(); sched.step()
        if (step+1)%10==0:
            row=dict(step=step+1,loss=float(loss.item()),owner=float(owner_loss.item()),transition=float(transition_loss.item()),
                     elapsed=previous_elapsed+time.monotonic()-tick,seconds_per_step=(time.monotonic()-tick)/(step+1-start))
            history.append(row); print(json.dumps(row),flush=True)
            dump(folder/"progress.json",dict(holdout=args.holdout,config=config,last=row,complete=False,history=history))
        if (step+1)%100==0 or step+1==config["steps"]:
            atomic_torch_save(torch,checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                step=step+1,config=config,source_identity=plan["source_identity"],numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None,
                holdout=args.holdout,role=args.role,inner_holdout=inner,training_code_sha256=code_sha256,
                train_games=[g for g in GAMES if g not in excluded],train_seeds=TRAIN_SEEDS))
    atomic_torch_save(torch,folder/"model.pt",dict(model=model.state_dict(),config=config,holdout=args.holdout,
        source_identity=plan["source_identity"],role=args.role,inner_holdout=inner,training_code_sha256=code_sha256,
        train_games=[g for g in GAMES if g not in excluded],train_seeds=TRAIN_SEEDS))
    dump(folder/"progress.json",dict(holdout=args.holdout,config=config,complete=True,history=history,
        elapsed=previous_elapsed+time.monotonic()-tick,torch=torch.__version__,device=str(device)))


def view_inputs(G,P,M,xf,yf):
    G,P,M=G.copy(),P.copy(),M.copy()
    if xf:
        G=G[..., [0,1,2,3,4,5,6,7,8,9,11,10,13,12,14,16,15,17,18,19]]
        G[..., [3,5]]*=-1
        P=np.flip(P,axis=-3).copy(); M=np.flip(M,axis=-2).copy(); P[..., [0,2,5]]*=-1
    if yf: G[..., [4,6]]*=-1; P[..., [1,3,6]]*=-1
    return G,P,M


def infer(torch,model,it,device,halo):
    owner=np.zeros((len(it["gt"]),3),np.float32); transition=np.zeros((len(it["gt"]),6),np.float32)
    direction_swap=[PAIRS.index((1-a if a<2 else a,1-b if b<2 else b)) for a,b in PAIRS]
    with torch.no_grad():
        for a,b in it["periods"]:
            for s in range(int(a),int(b),4096):
                e=min(int(b),s+4096); lo=max(int(a),s-halo); hi=min(int(b),e+halo)
                arrays=[np.array(it[name][lo:hi],dtype=bool if name=="M" else np.float32) for name in ("G","P","M")]
                op,tp=[],[]
                for xf,yf in ((0,0),(1,0),(0,1),(1,1)):
                    inputs=[torch.from_numpy(v)[None].to(device) for v in view_inputs(*arrays,xf,yf)]
                    own,tr=model(*inputs)
                    own=torch.softmax(own[0,s-lo:e-lo],dim=1).cpu().numpy()
                    tr=torch.sigmoid(tr[0,s-lo:e-lo]).cpu().numpy()
                    if xf: own=own[:,[1,0,2]]; tr=tr[:,direction_swap]
                    op.append(own); tp.append(tr)
                owner[s:e]=np.log(np.clip(np.mean(op,axis=0),1e-7,1))
                q=np.clip(np.mean(tp,axis=0),1e-7,1-1e-7)
                transition[s:e]=np.log(q/(1-q))
    return owner,transition


def predict(args):
    torch=torch_setup(args.threads); device=torch.device(args.device)
    plan=ensure_plan(args.out); games=[args.holdout] if args.holdout else GAMES
    for outer in games:
        ck=torch.load(args.out/"folds"/outer/args.role/"model.pt",map_location=device,weights_only=False)
        game=ck["inner_holdout"] if args.role=="inner" else outer
        assert game not in ck["train_games"] and outer not in ck["train_games"] and ck["holdout"]==outer
        assert ck["source_identity"]==plan["source_identity"]
        assert ck["training_code_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        model=make_net(torch,ck["config"]).to(device); model.load_state_dict(ck["model"]); model.eval()
        for seed in ([104] if args.role=="inner" else args.seeds):
            stem=f"{outer}-inner-{game}-{seed}" if args.role=="inner" else f"{game}-{seed}"
            destination=args.out/"probabilities"/f"{stem}.npz"
            if destination.exists():
                with np.load(destination) as saved:
                    assert str(saved["source_identity"])==plan["source_identity"]
                    assert str(saved["training_code_sha256"])==ck["training_code_sha256"]
                    assert str(saved["outer_holdout"])==outer and str(saved["evaluation_game"])==game
                    assert str(saved["role"])==args.role and int(saved["seed"])==seed
                continue
            it=read_item(args.out/"cache"/f"{game}-{seed}")
            owner,transition=infer(torch,model,it,device,ck["config"]["halo"])
            assert np.isfinite(owner).all() and np.isfinite(transition).all()
            destination.parent.mkdir(parents=True,exist_ok=True)
            tmp=destination.with_suffix(".tmp")
            with tmp.open("wb") as f: np.savez_compressed(f,owner=owner,transition=transition,
                source_identity=plan["source_identity"],outer_holdout=outer,evaluation_game=game,role=args.role,
                training_code_sha256=ck["training_code_sha256"],seed=seed)
            tmp.replace(destination); print("predicted",outer,args.role,game,seed,flush=True)


def decode_grid(owner,evidence,periods,settings):
    """Batched dense Viterbi settings=(HAcost,LDcost,directionweight). No peaks/duration gate."""
    n=len(owner); count=len(settings); pred=np.full((n,count),3,np.uint8)
    costs=np.array([[[0,ha,ld],[ha,0,ld],[ld,ld,0]] for ha,ld,w in settings],np.float32)
    weights=np.array([w for ha,ld,w in settings],np.float32)
    evidence=np.clip(evidence,-8,8)
    template=np.zeros((n,3,3),np.float32)
    for k,(i,j) in enumerate(PAIRS): template[:,i,j]=evidence[:,k]
    fixed=-costs
    fixed=fixed-np.log(np.exp(fixed).sum(2,keepdims=True))
    dynamic=bool(np.any(weights))
    batchindex=np.arange(count)[:,None]; states=np.arange(3)[None,:]
    for a,b in periods:
        a,b=int(a),int(b); back=np.zeros((b-a,count,3),np.int8)
        value=np.broadcast_to(owner[a],(count,3)).astype(np.float64).copy()
        for t in range(a+1,b):
            if dynamic:
                logp=-costs+weights[:,None,None]*template[t]
                logp-=logp.max(2,keepdims=True)
                logp-=np.log(np.exp(logp).sum(2,keepdims=True))
            else: logp=fixed
            score=value[:,:,None]+logp
            back[t-a]=score.argmax(1)
            value=score[batchindex,back[t-a],states]+owner[t]
        k=value.argmax(1); pred[b-1]=k
        for t in range(b-1,a,-1): k=back[t-a,np.arange(count),k]; pred[t-1]=k
    return pred


def calibrated_evidence(transition,calibration):
    prior=np.clip(np.array(calibration["prior"]),1e-6,1-1e-6)
    return transition*np.array(calibration["slope"])+np.array(calibration["bias"])-np.log(prior/(1-prior))


def calibrate(args):
    """One inner whole match per outer fold; no outer model/logit/label enters calibration."""
    sys.path.insert(0,str(ROOT/"evaluation")); import possession_metrics as pm
    torch=torch_setup(args.threads); device=torch.device(args.device)
    games=[args.holdout] if args.holdout else GAMES; plan=ensure_plan(args.out)
    for outer in games:
        destination=args.out/"folds"/outer/"calibration.json"
        if destination.exists():
            saved=json.loads(destination.read_text())
            assert saved["source_identity"]==plan["source_identity"] and saved["outer_holdout"]==outer
            assert saved["training_code_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            continue
        inner=GAMES[(GAMES.index(outer)+1)%len(GAMES)]
        it=read_item(args.out/"cache"/f"{inner}-104")
        with np.load(args.out/"probabilities"/f"{outer}-inner-{inner}-104.npz") as z:
            assert str(z["outer_holdout"])==outer and str(z["evaluation_game"])==inner and str(z["role"])=="inner"
            assert int(z["seed"])==104 and str(z["source_identity"])==plan["source_identity"]
            assert str(z["training_code_sha256"])==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            owner=z["owner"].copy(); transition=z["transition"].copy()
        target=directed_targets(it["gt"],it["periods"])
        valid=transition_valid(it["gt"],it["periods"])
        x=torch.from_numpy(transition[valid]).to(device); y=torch.from_numpy(target[valid].astype(np.float32)).to(device)
        raw=torch.nn.Parameter(torch.full((6,),.5413248546,device=device))
        bias=torch.nn.Parameter(torch.full((6,),-np.log(CONFIG["positive_weight"]),device=device,dtype=torch.float32))
        optimiser=torch.optim.LBFGS([raw,bias],lr=.5,max_iter=100,line_search_fn="strong_wolfe")
        def closure():
            optimiser.zero_grad(); slope=torch.nn.functional.softplus(raw)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(x*slope+bias,y)
            loss=loss+.001*((slope-1)**2).mean()+.00001*(bias**2).mean()
            loss.backward(); return loss
        optimiser.step(closure)
        fitted=dict(slope=torch.nn.functional.softplus(raw).detach().cpu().numpy().tolist(),
            bias=bias.detach().cpu().numpy().tolist(),prior=target[valid].mean(0).tolist())
        evidence=calibrated_evidence(transition,fitted)
        grid=[(ha,ld,0.) for ha in (1,2,4,8,16) for ld in (2,4,8,16,32)]
        preds=decode_grid(owner,evidence,it["periods"],grid)
        runs=pm.valid_runs(it["gt"])
        scoring=lambda pred: float(pm.spell_scores(it["gt"],pm.fill_abstentions(pred,runs),runs)["mean_f1"])
        scores=[scoring(preds[:,i]) for i in range(len(grid))]
        winner=int(np.argmax(scores)); ha,ld,_=grid[winner]
        directed_grid=[(ha,ld,w) for w in (0.,.25,.5,1.)]
        preds=decode_grid(owner,evidence,it["periods"],directed_grid)
        directed_scores=[scoring(preds[:,i]) for i in range(len(directed_grid))]
        directed_winner=int(np.argmax(directed_scores))
        fitted.update(outer_holdout=outer,inner_holdout=inner,calibration_seed=104,
            train_games=[g for g in GAMES if g not in (outer,inner)],train_seeds=TRAIN_SEEDS,
            source_identity=plan["source_identity"],training_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            owner_setting=grid[winner],directed_setting=directed_grid[directed_winner],
            owner_grid=[dict(setting=s,spell_f1=v) for s,v in zip(grid,scores)],
            directed_grid=[dict(setting=s,spell_f1=v) for s,v in zip(directed_grid,directed_scores)],
            limitation="Head calibration learned from five-game inner model; applied to architecture-identical six-game outer model. Cyclic inner assignment can cross DFL/Metrica label domains. No outer match exposure.",
            evidence_clip=8)
        dump(destination,fitted)
        print(json.dumps(dict(outer=outer,inner=inner,owner=fitted["owner_setting"],directed=fitted["directed_setting"],
            owner_inner_score=scores[winner],directed_inner_score=directed_scores[directed_winner])),flush=True)


def decode(owner,transition,periods,directed,calibration=None):
    if calibration is None:
        evidence=np.zeros_like(transition); setting=(2,4,0)
    else:
        evidence=calibrated_evidence(transition,calibration)
        setting=calibration["directed_setting" if directed else "owner_setting"]
    return decode_grid(owner,evidence,periods,[setting])[:,0]


def evaluate(args):
    sys.path.insert(0,str(ROOT))
    from common.rapid_exchanges import score, dump as json_dump
    rows=[]; plan=ensure_plan(args.out); code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for game in GAMES:
        calibration=json.loads((args.out/"folds"/game/"calibration.json").read_text())
        assert calibration["outer_holdout"]==game and game not in calibration["train_games"]
        assert calibration["source_identity"]==plan["source_identity"] and calibration["training_code_sha256"]==code_sha256
        for seed in args.seeds:
            it=read_item(args.out/"cache"/f"{game}-{seed}")
            with np.load(args.out/"probabilities"/f"{game}-{seed}.npz") as p:
                assert str(p["source_identity"])==plan["source_identity"] and str(p["training_code_sha256"])==code_sha256
                assert str(p["role"])=="main" and str(p["outer_holdout"])==game and str(p["evaluation_game"])==game and int(p["seed"])==seed
                row=dict(game=game,seed=seed)
                for label,directed in (("owner",False),("directed",True)):
                    pred=decode(p["owner"],p["transition"],it["periods"],directed,calibration)
                    row[label]=score(it["gt"],pred,25)
                    path=args.out/"predictions"/f"{game}-{seed}-{label}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,prediction=pred)
                rows.append(row); json_dump(args.out/"scores.json",dict(complete=False,rows=rows))
                print(json.dumps(dict(game=game,seed=seed,owner=row["owner"]["spell_f1"],directed=row["directed"]["spell_f1"])),flush=True)
    json_dump(args.out/"scores.json",dict(complete=True,rows=rows,
        means={k:float(np.mean([r[k]["spell_f1"] for r in rows])) for k in ("owner","directed")}))


def check(args):
    # Directed heads retain opposite adjacent events even when their tolerance blobs overlap.
    y=np.array([0]*10+[1]*2+[0]*10+[3]+[1]*10+[0]*10,np.uint8)
    target=directed_targets(y,[(0,22),(23,43)])
    assert target[10,0] and target[12,2] and not target[22].any() and not target[23].any()
    valid=transition_valid(y,[(0,22),(23,43)])
    assert not valid[[0,22,23]].any() and valid[24]
    rng=np.random.default_rng(4)
    arrays=(rng.normal(size=(2,30,20)).astype(np.float32),rng.normal(size=(2,30,2,12,18)).astype(np.float32),
            rng.random((2,30,2,12))>.5)
    for xf,yf in ((0,0),(1,0),(0,1),(1,1)):
        actual=view_inputs(*view_inputs(*arrays,xf,yf),xf,yf)
        assert all(np.array_equal(a,b) for a,b in zip(arrays,actual)),"symmetry view is not involutive"
    own=np.full((40,3),-10.,np.float32); own[:,0]=0; own[15:27,1]=2
    tr=np.full((40,6),-20.,np.float32)
    decoded=decode(own,tr,[(0,40)],False)
    assert (decoded[15:27]==1).all() and (decoded[:15]==0).all()
    torch=torch_setup(args.threads)
    model=make_net(torch,CONFIG)
    G=torch.zeros(2,300,20); P=torch.zeros(2,300,2,11,18); M=torch.zeros(2,300,2,11,dtype=torch.bool)
    own,tr=model(G,P,M); assert own.shape==(2,300,3) and tr.shape==(2,300,6)
    assert torch.isfinite(own).all() and torch.isfinite(tr).all()
    print(json.dumps(dict(checks="passed",torch=torch.__version__,parameters=sum(p.numel() for p in model.parameters()))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("build","train","predict","calibrate","evaluate","check"))
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--holdout",choices=GAMES)
    ap.add_argument("--role",choices=("main","inner"),default="main")
    ap.add_argument("--games",nargs="+")
    ap.add_argument("--seeds",nargs="+",type=int,default=list(TRAIN_SEEDS+EVAL_SEEDS))
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--steps",type=int)
    ap.add_argument("--batch",type=int)
    args=ap.parse_args()
    globals()[args.stage](args)


if __name__=="__main__": main()
