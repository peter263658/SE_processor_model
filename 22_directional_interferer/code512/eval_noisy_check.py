import sys, glob, os, re, numpy as np, soundfile as sf, warnings
warnings.simplefilter("ignore")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from MBSTOI.mbstoi import mbstoi
from concurrent.futures import ProcessPoolExecutor
D="/raid/R12K41024/BCCTN/Dataset"
rx=re.compile(r'_snr[-+]?[\d.]+\.wav$')
def one(f):
    cl=os.path.join(f"{D}/clean_testset",rx.sub(".wav",os.path.basename(f)))
    if not os.path.exists(cl): return None
    n,_=sf.read(f,dtype="float32",always_2d=True); c,_=sf.read(cl,dtype="float32",always_2d=True)
    L=c.shape[0]
    try:
        v=mbstoi(c[:,0],c[:,1],n[:L,0],n[:L,1],gridcoarseness=1)
        return v if np.isfinite(v) else None
    except: return None
for cond,root in (("dirnoise90",f"{D}/noisy_testset_dirnoise90"),("isotropic",f"{D}/noisy_testset")):
    allv=[]
    for s in [-6,-3,0,3,6,9,12,15]:
        fs=sorted(glob.glob(f"{root}/snr_{s}dB/*.wav"))[:750]
        with ProcessPoolExecutor(max_workers=32) as ex:
            allv+=[v for v in ex.map(one,fs) if v is not None]
    print(f"{cond} noisy MBSTOI (eval_min 路徑) = {np.mean(allv):.4f}  n={len(allv)}")
