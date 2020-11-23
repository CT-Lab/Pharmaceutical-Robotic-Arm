# RL with Dopamine and Webots
## How to run it?
* RL Agent: Run <code>python3 train.py --agent_name=ddpg --base_dir=/tmp/webots --gin_files='agents/ddpg/configs/ddpg.gin'</code>.
  * Change ***--base_dir=/tmp/webots*** for differnt neural networks.
  * For example, <code>python3 train.py --agent_name=ddpg --base_dir=/home/sctech/CS/Pharmaceutical-Robotic-Arm/backup_down_version/webots  --gin_files='agents/ddpg/configs/ddpg.gin'</code>
* Webots: Open webots with /Dopamine_Webots/XXX.wbt.
## Current Achievements and work
### Achievements: Move Down/Up and Reach a Target via DDPGAgent
<table>
  <tr>
    <td align="center"><b>Move Down</b><br /><sub><img src="https://github.com/CT-Lab/Pharmaceutical-Robotic-Arm/blob/Dopamine_Webots/img/20201123_190054.gif" width="400px;" alt=""/></sub><br /></td>
   <td align="center"><b>Move Up</b><br /><sub><img src="https://github.com/CT-Lab/Pharmaceutical-Robotic-Arm/blob/Dopamine_Webots/img/20201123_200537.gif" width="400px;" alt=""/></sub><br /></td>
  </tr>
</table>
### Trend
![image](https://github.com/CT-Lab/Pharmaceutical-Robotic-Arm/blob/Dopamine_Webots/img/%E5%9C%96%E7%89%873.png)
