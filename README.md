# RL with Dopamine and Webots
## How to run it?
* RL Agent: Run <code>python3 train.py --agent_name=ddpg --base_dir=/tmp/webots --gin_files='agents/ddpg/configs/ddpg.gin'</code>.
  * Change ***--base_dir=/tmp/webots*** for differnt neural networks.
  * For example, <code>python3 train.py --agent_name=ddpg --base_dir=/home/sctech/CS/Pharmaceutical-Robotic-Arm/backup_down_version/webots  --gin_files='agents/ddpg/configs/ddpg.gin'</code>
* Webots: Open webots with /Dopamine_Webots/XXX.wbt.
## Current Achievements and work
### Achievements: Move Down and Reach a Target via DDPGAgent
<center class="half">
    <img src="https://img-blog.csdn.net/2018061215200776?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="200"/><img src="https://img-blog.csdn.net/20180612152032532?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="200"/><img src="https://img-blog.csdn.net/20180612152100203?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxODA4OTYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="200"/>
</center>
<center class="half">
    <img src="https://github.com/CT-Lab/Pharmaceutical-Robotic-Arm/blob/Dopamine_Webots/img/20201123_190054.gif" width="200"/><img src="https://github.com/CT-Lab/Pharmaceutical-Robotic-Arm/blob/Dopamine_Webots/img/20201123_190054.gif" width="200"/>
</center>
