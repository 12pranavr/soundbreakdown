import numpy as np, soundfile as sf
sr=44100; bpm=120; beat=60/bpm; bar=4*beat; rng=np.random.default_rng(0)
def env(n,a=0.005,d=0.2):
    t=np.arange(n)/sr; return np.minimum(t/a,1)*np.exp(-t/d)
def kick():
    n=int(.4*sr); t=np.arange(n)/sr; f=50+100*np.exp(-t*30); return np.sin(2*np.pi*np.cumsum(f)/sr)*env(n,.001,.15)
def snare():
    n=int(.25*sr); return (rng.standard_normal(n)*.7+np.sin(2*np.pi*190*np.arange(n)/sr))*env(n,.001,.07)
def hat():
    n=int(.06*sr); x=rng.standard_normal(n); x=np.diff(x,prepend=0); return x*env(n,.0005,.015)*.5
def tone(f,dur,kind="saw",d=1.0):
    n=int(dur*sr); t=np.arange(n)/sr
    x=sum(np.sin(2*np.pi*f*k*t)/k for k in range(1,12)) if kind=="saw" else np.sin(2*np.pi*f*t)
    return x*env(n,.01,d)
nbars=40; N=int(nbars*bar*sr)+sr; L=np.zeros((2,N))
def add(x,t,pan=0.0,g=1.0):
    i=int(t*sr); x=x[:N-i]; L[0,i:i+len(x)]+=x*g*(1-pan); L[1,i:i+len(x)]+=x*g*(1+pan)
prog=[(220,"Am"),(174.6,"F"),(130.8,"C"),(196,"G")]
chords={220:[220,261.6,329.6],174.6:[174.6,220,261.6],130.8:[261.6,329.6,392],196:[196,246.9,293.7]}
for b in range(nbars):
    t0=b*bar; root=prog[b%4][0]
    sec = "intro" if b<8 else "verse" if b<16 else "chorus" if b<24 else "break" if b<32 else "chorus"
    for f in chords[root]:  # pad
        add(tone(f,bar,"sine",3.0),t0,pan=0.4 if f%2 else -0.4,g=0.08 if sec!="chorus" else 0.12)
    if sec in("verse","chorus"):
        for s in (0,2.5): add(kick(),t0+s*beat,g=0.9)
        for s in (1,3): add(snare(),t0+s*beat,g=0.5)
        for s in range(8): add(hat(),t0+s*beat/2,pan=0.3,g=0.6)
        for s in (0,1.5,2.5,3): add(tone(root/4,beat*.45,"saw",.25),t0+s*beat,g=0.25)
    if sec=="chorus":
        mel=[root*2,root*2*1.2,root*3,root*2*1.5]
        for s,f in enumerate(mel): add(tone(f,beat*.9,"saw",.5),t0+s*beat,pan=-0.5,g=0.08)
L/=np.abs(L).max()*1.1
sf.write("test/test_song.wav",L.T,sr)
print("ok", N/sr)
