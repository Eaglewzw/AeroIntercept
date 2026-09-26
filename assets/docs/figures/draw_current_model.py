"""Publication-style vector diagram of the currently trained two-frame policy."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','svg.fonttype':'path','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(16,7.5))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,1600),ylim=(750,0));ax.axis('off')
ink='#202832';muted='#58616c';blue='#365b7e';amber='#997341'
def txt(x,y,s,size=11,color=ink,weight='normal',ha='center'):
 ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,weight=weight,linespacing=1.5)
def box(x,y,w,h,title,detail,fill='#f3f6f9',edge=blue,dash=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=4',facecolor=fill,edgecolor=edge,linewidth=1.1,linestyle='--' if dash else '-'))
 txt(x+w/2,y+24,title,11.5,weight='bold');txt(x+w/2,y+(h+30)/2,detail,10)
def arrow(points,color=ink,dash=False):
 ax.add_patch(FancyArrowPatch(path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)),arrowstyle='-|>',mutation_scale=12,linewidth=1.25,color=color,linestyle='--' if dash else '-'))
# Solid: inference data flow. Dashed: auxiliary training branch.
txt(42,35,'AeroIntercept: visual policy with proprioceptive fusion',20,weight='bold',ha='left')
txt(42,83,'(a) Actor — inference',13,blue,'bold','left')
box(40,228,175,104,'Visual observation',r'$I_{t-1}, I_t$'+'\n'+r'$2\times3\times640\times640$'+'\nFull-frame letterbox')
box(260,228,235,104,'Shared visual encoder','ResNet-18 · multi-scale\nSpatial attention pooling\n+ spatial coordinates')
box(540,228,220,104,'Temporal encoder','2-layer Transformer\n2 frame tokens\n'+r'$z_t^{v}\in\mathbb{R}^{128}$')
box(805,228,205,104,'Feature fusion','Concatenate + linear\n'+r'$128+64\rightarrow128$'+'\n'+r'$z_t\in\mathbb{R}^{128}$')
box(1055,228,225,104,'Action head',r'MLP $128\rightarrow128\rightarrow4$'+'\nYaw feedback addition\n'+r'$\tanh(\cdot)$')
box(1325,228,235,104,'Control command',r'$(v_f,v_r,v_d,\dot{\psi})$'+'\nScale / norm limit\nFRD → NED → PX4',fill='#fafafa',edge=muted)
for a,b in [(215,260),(495,540),(760,805),(1010,1055),(1280,1325)]:arrow([(a,280),(b,280)])
box(610,108,415,74,'Image-based yaw feedback',r'$b_t^{\psi}=k_\psi\arctan(\bar{x}_t\tan(\theta_h/2))$',fill='#fbf8f2',edge=amber)
arrow([(377,228),(377,145),(610,145)],amber)
arrow([(1025,145),(1167,145),(1167,228)],amber)
txt(390,187,'Current-frame attention',9,amber,ha='left')
box(260,411,235,95,'Proprioceptive state',r'$s_t=[v_x,v_y,v_z,\omega_x,\omega_y,\omega_z]$'+'\nPX4 body-frame measurements')
box(540,411,220,95,'State encoder','Physical-scale normalization\n'+r'MLP $6\rightarrow64\rightarrow64$')
arrow([(495,458),(540,458)])
arrow([(760,458),(907,458),(907,332)])
box(1055,411,360,95,'Auxiliary prediction heads','Future position (3) · risk (1)\nVisibility (1)',fill='#fafafa',edge=muted,dash=True)
arrow([(1010,313),(1033,313),(1033,458),(1055,458)],muted,True)
# Privileged critic is independent of the deployed actor.
ax.add_patch(FancyBboxPatch((40,560),1520,110,boxstyle='round,pad=0,rounding_size=4',facecolor='#fafafa',edgecolor='#a9afb6',linewidth=1,linestyle='--'))
txt(60,582,'(b) Privileged critic — PPO training only',12,muted,'bold','left')
txt(265,626,r'Simulator state $s_t^{priv}\in\mathbb{R}^{15}$',12)
arrow([(475,626),(610,626)],muted)
txt(840,626,r'MLP $15\rightarrow256\rightarrow256\rightarrow1$',12)
arrow([(1070,626),(1220,626)],muted)
txt(1380,626,r'State value $V(s_t^{priv})$',12)
txt(42,702,'Current checkpoint: two-frame BC policy; recurrent memory disabled. Auxiliary branches provide training supervision.',10,muted,ha='left')
txt(42,728,'Target ground truth is excluded from the Actor. The PPO critic is not used during inference or the current BC training.',10,muted,ha='left')
for ext in ['png','svg','pdf']:
 p=OUT/f'current_model_architecture.{ext}';fig.savefig(p,dpi=220,facecolor='white');print(p)
plt.close(fig)
p=OUT/'current_model_architecture.svg'
p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
