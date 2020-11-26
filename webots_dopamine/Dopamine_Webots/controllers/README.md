# IMPORTANT: new comments are started with "##"

How to execute the program

run simulation in Webots and it will create pipe to transport information

python3 train.py --agent_name=ddpg --base_dir=/tmp/webots --gin_files='agents/ddpg/configs/ddpg.gin'

python3 endpoint_evl.py

# For demo (up + down)

python3 eval_up.py --agent_name=ddpg --schedule='eval' --base_dir=/home/sctech/webots_dopamine/up/webots --gin_files='agents/ddpg/configs/ddpg.gin'

python3 eval_down.py --agent_name=ddpg --schedule='eval' --base_dir=/home/sctech/webots_dopamine/down/webots --gin_files='agents/ddpg/configs/ddpg_down.gin'
