import asyncio
import time
import json
import sys
import re
from enum import Enum
sys.path.append("../")
from ray.rllib.models import ModelCatalog
from agi.nl_holdem_env import NlHoldemEnvWrapper
from agi.nl_holdem_net import NlHoldemNet
ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
import numpy as np
from tqdm import tqdm
import pandas as pd
from agi.evaluation_tools import NNAgent,death_match

import tensorflow as tf

#%%

conf = eval(open("../confs/nl_holdem.py").read().strip())

#%%

env = NlHoldemEnvWrapper(
        conf
)

#%%
i = 441
nn_agent = NNAgent(env.observation_space,
                       env.action_space,
                       conf,
                       f"../weights/c_{i}.pkl",
                       f"oppo_c{i}")

color2ind = dict(zip("CDHS",[0,1,2,3]))
rank2ind = dict(zip("23456789TJQKA",[0,1,2,3,4,5,6,7,8,9,10,11,12]))

#%%

for i in tqdm(range(1)):
    obs = env.reset()
    
    
    d = False
    while not d:
        action_ind = nn_agent.make_action(obs)
        obs,r,d,i = env.step(action_ind)

    
        # print(obs.keys())
        # print('card_info:\n')
        # print(obs['card_info'])
        # print('action_info:\n')
        # print(obs['action_info'])
        # print('legal_moves:\n')
        # print(obs['legal_moves'])
        # print('extra_info:\n')
        # print(obs['extra_info'])
    #break

#%%



print(
    env.env.get_state(0)["raw_obs"]["hand"],\
    env.env.get_state(1)["raw_obs"]["hand"],\
    env.env.get_state(1)["raw_obs"]["public_cards"],\
    env.env.get_state(1)["action_record"],
    env.history
    )


# #%%

# print(env.env.get_payoffs())


#%%

class Action(Enum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    ALL_IN = 4
    # Newly added actions
    RAISE_ONETHIRD_POT = 5
    RAISE_THREEFOURTH_POT = 6
    RAISE_ONEANDHALF_POT = 7
    RAISE_TWO_POT = 8
    RAISE_THREE_POT = 9


def create_obs(obs_dict=None):
    """
    Create a valid observation object for the poker environment.
    
    Args:
        obs_dict (dict): Dictionary containing observation parameters:
            {
                "hand_cards": list of hole cards e.g. ['AS', 'KH'],
                "public_cards": list of community cards e.g. ['JD', 'TD', '3C'],
                "history": list of actions for each stage,
                "legal_actions": list of legal action indices,
                "stakes": tuple of (player_stake, opponent_stake),
                "current_player": current player ID (0 or 1)
            }
            
    Returns:
        dict: Observation dictionary with required structure
    """
    if obs_dict is None:
        obs_dict = {}

    # Extract parameters from dictionary with defaults
    hand_cards = obs_dict.get('hand_cards', None)
    public_cards = obs_dict.get('public_cards', None)
    history = obs_dict.get('history', None)
    legal_actions = obs_dict.get('legal_actions', None)
    stakes = obs_dict.get('stakes', None)
    current_player = obs_dict.get('current_player', 0)

    if hand_cards is None:
        hand_cards = []
    if public_cards is None:
        public_cards = []
    if legal_actions is None:
        legal_actions = list(range(10))  # All actions legal by default
    if stakes is None:
        stakes = (0, 0)
        
    # Initialize observation components
    card_info = np.zeros([4, 13, 6], np.uint8)
    action_info = np.zeros([4, 10, 25], np.uint8)  # Increased to 10 actions, kept third dimension at 25
    extra_info = np.zeros([2], np.uint8)
    legal_actions_info = np.zeros([10], np.uint8)  # Increased to match new action count
    
    # Set legal actions
    for ind in legal_actions:
        legal_actions_info[ind] = 1
    
    # Process cards
    flop_cards = public_cards[:3]
    turn_cards = public_cards[3:4]
    river_cards = public_cards[4:5]
    
    # Process hole cards
    for card in hand_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][0] = 1
                
    # Process flop cards
    for card in flop_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][1] = 1
                
    # Process turn card
    for card in turn_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][2] = 1
                
    # Process river card
    for card in river_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][3] = 1
                
    # Process all public cards
    for card in public_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][4] = 1
                
    # Process all visible cards
    for card in public_cards + hand_cards:
        if len(card) == 2:
            color, rank = card[0], card[1]
            if color in color2ind and rank in rank2ind:
                card_info[color2ind[color]][rank2ind[rank]][5] = 1
    
    # Process action Info 
    if history:
        for ind_round, one_history in enumerate(history):
            print('ind_round', ind_round)
            print('one_history', one_history)
            for ind_h, (player_id, action_id, legal_actions) in enumerate(one_history[:6]):
                action_info[player_id, action_id, ind_round * 6 + ind_h] = 1
                action_info[2, action_id, ind_round * 6 + ind_h] = 1
                
                for la_ind in legal_actions:
                    action_info[3, la_ind, ind_round * 6 + ind_h] = 1
    
    # Set current player in action info
    action_info[:, :, -1] = current_player
    
    # Set stakes info
    extra_info[0] = stakes[0]
    extra_info[1] = stakes[1]
    
    return {
        "card_info": card_info,
        "action_info": action_info,
        "legal_moves": legal_actions_info,
        "extra_info": extra_info
    }

def create_obs_from_env_state(env):
    """从环境状态创建observation对象
    
    Args:
        env: 游戏环境实例
        
    Returns:
        dict: observation对象
    """
    # 从env.env获取状态信息
    obs=env.last_obs
    hand_cards = obs[0]["raw_obs"]["hand"]
    print('hand_cards:', hand_cards)
    public_cards = obs[0]["raw_obs"]["public_cards"]
    print('public_cards:', public_cards)
    current_legal_actions = [i.value for i in obs[0]["raw_obs"]["legal_actions"]]
    print('current_legal_actions:', current_legal_actions)
    # 获取动作历史
    action_record = obs[0]["action_record"]
    print('action record:', action_record)
    # 将action_record转换为history格式
    history = [[], [], [], []]  # preflop, flop, turn, river
    current_stage = 0
    
    for action in action_record:
        # 假设action_record中的每个动作是一个tuple (player_id, action_type)
        player_id, action_type = action
        
        # 添加到history
        history[current_stage].append([
            player_id,
            action_type,
            [0,1,2,3,4,5,6,7,8,9]  # 所有可能的动作
        ])
    print('history:', history)
    print('env_history:', env.history)
    # 获取当前合法动作
    legal_actions = [action.value for action in obs[0]["raw_obs"]["legal_actions"]]
    print('legal_actions:', legal_actions)
    # 获取双方筹码
    stakes = obs[0]["raw_obs"]["stakes"]
    print('stakes:', stakes)
    # 获取当前玩家
    current_player = obs[0]["raw_obs"]["current_player"] if "current_player" in obs[0]["raw_obs"] else 0
    print('current player:', current_player)
    
    # 创建observation字典
    obs_dict = {
        'hand_cards': hand_cards,
        'public_cards': public_cards,
        'history': env.history,
        'legal_actions': legal_actions,
        'stakes': stakes,
        'current_player': current_player
    }
    
    # 使用新的obs_dict格式创建observation
    return create_obs(obs_dict)

def test_obs_creation():
    """测试observation创建函数，逐个比较observation的各个部分"""
    import numpy as np
    
    # 创建一个环境实例
    env = NlHoldemEnvWrapper(
        conf
    )
    
    # 获取一个正常的observation作为参考
    for i in tqdm(range(2)):
        original_obs = env.reset()
        d = False
        while not d:
            action_ind = nn_agent.make_action(original_obs)
            original_obs,r,d,i = env.step(action_ind)
            print('legal_moves', original_obs['legal_moves'])
            # 使用环境状态创建新的observation
            new_obs = create_obs_from_env_state(env)
            
            # 逐个比较各个部分
            components = ["card_info", "action_info", "legal_moves", "extra_info"]
            
            for comp in components:
                print(f"\nComparing {comp}:")
                print(f"Original shape: {original_obs[comp].shape}")
                print(f"New shape: {new_obs[comp].shape}")
                
                if original_obs[comp].shape != new_obs[comp].shape:
                    print(f"Shape mismatch in {comp}!")
                    continue
                    
                # 检查数值是否相等
                is_equal = np.array_equal(original_obs[comp], new_obs[comp])
                print(f"Values are equal: {is_equal}")
                
                if not is_equal:
                    # 找出不同的位置
                    diff_indices = np.where(original_obs[comp] != new_obs[comp])
                    print(f"Differences found at indices: {diff_indices}")
                    print("Original values at these positions:", original_obs[comp][diff_indices])
                    print("New values at these positions:", new_obs[comp][diff_indices])
                    
                    # 如果是多维数组，打印第一个不同的位置的完整切片
                    if len(diff_indices[0]) > 0:
                        first_diff = tuple(index[0] for index in diff_indices)
                        print(f"\nFirst difference at index {first_diff}")
                        if len(original_obs[comp].shape) > 1:
                            print("Original slice:")
                            print(original_obs[comp][first_diff[0]])
                            print("New slice:")
                            print(new_obs[comp][first_diff[0]])
        #break
    
    
    return original_obs, new_obs






#*************************
# obs = create_obs(
#     hand_cards=['SA', 'HK'],  # 黑桃A和红心K
#     public_cards=['DJ', 'DT', 'C3'],  # 公共牌：方块J、方块10、梅花3
#     legal_actions=[0, 1, 2],  # 可用动作
#     stakes=(10, 20),  # 双方赌注
#     current_player=0  # 当前玩家
# )
# print('card_info:')
# print(obs['card_info'])
# print('action_info:')
# print(obs['action_info'])
# print('legal_moves:')
# print(obs['legal_moves'])
# print('extra_info:')
# print(obs['extra_info'])
#*************************

def convert_card_format(card):
    """将ACPC格式的牌转换为我们的格式
    
    Args:
        card: ACPC格式的牌，例如'Ah'（A♥），格式是[rank][suit]
    
    Returns:
        str: 我们的格式，例如'HA'（♥A），格式是[suit][rank]
    """
    # ACPC格式: [rank][suit]，例如'Ah'表示A♥
    # 我们的格式: [suit][rank]，例如'HA'表示♥A
    rank_map = {'t': '10', 'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A'}
    
    if not card or len(card) != 2:
        return None
        
    rank, suit = card[0], card[1]
    
    # 转换rank
    if rank.lower() in rank_map:
        rank = rank_map[rank.lower()]
    else:
        rank = rank.upper()
    
    # 转换suit
    suit = suit.upper()
    
    # 返回[suit][rank]格式
    return suit + rank


async def handle_connection(reader, writer):
    """处理单个连接的协程，接收obs_dict结构的json string，返回raise指令
    
    Args:
        reader: StreamReader对象，用于读取数据
        writer: StreamWriter对象，用于发送数据
    """
    addr = writer.get_extra_info('peername')
    print(f"New connection from {addr}")
    
    try:
        while True:
            data = await reader.readline()
            if not data:
                print(f"Connection closed by {addr}")
                break
            try:
                # 解析接收到的JSON数据
                obs_dict = json.loads(data.decode().strip())
                print(f"Received from {addr}: {obs_dict}")
                
                # 验证obs_dict结构
                required_keys = ['hand_cards', 'public_cards', 'history', 'legal_actions', 'stakes', 'current_player']
                if not all(key in obs_dict for key in required_keys):
                    raise ValueError("Missing required keys in obs_dict")
                
                # 创建observation
                print(obs_dict)
                obs = create_obs(obs_dict)
                #print(f"Created observation: {obs}")
                
                # 获取动作索引并转换为Action枚举
                action_ind = nn_agent.make_action(obs)
                action = Action(action_ind)  # 将整数转换为Action枚举
                print(f"action: {action.name} ({action.value})")
                writer.write(action.name.encode() + b'\n')
                await writer.drain()
                
            except json.JSONDecodeError:
                error_msg = "Invalid JSON format"
                writer.write(f"{{'error': '{error_msg}'}}".encode() + b'\n')
                await writer.drain()
            except ValueError as e:
                error_msg = str(e)
                writer.write(f"{{'error': '{error_msg}'}}".encode() + b'\n')
                await writer.drain()
            except Exception as e:
                error_msg = f"Server error: {str(e)}"
                writer.write(f"{{'error': '{error_msg}'}}".encode() + b'\n')
                await writer.drain()
                
    except Exception as e:
        print(f"Connection error with {addr}: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"Connection to {addr} closed")


async def run_server():
    """运行服务器"""
    server = await asyncio.start_server(
        handle_connection, 
        '0.0.0.0',  # 监听所有网络接口
        8888        # 端口号
    )

    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    async with server:
        await server.serve_forever()


def start_server():
    """启动服务器的入口函数"""
    try:
        # 在Windows上需要使用不同的事件循环策略
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 运行服务器
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")


async def test_client(obs_dict):
    """测试客户端
    
    Args:
        obs_dict: 观察字典，包含游戏状态信息
    """
    try:
        reader, writer = await asyncio.open_connection('127.0.0.1', 8888)
        
        # 发送obs_dict
        json_str = json.dumps(obs_dict)
        writer.write(json_str.encode() + b'\n')
        await writer.drain()
        
        # 接收响应
        response = await reader.readline()
        if response:
            print(f"Server response: {response.decode().strip()}")
            
        writer.close()
        await writer.wait_closed()
            
    except Exception as e:
        print(f"Client error: {e}")


def test_server():
    """测试服务器"""
    # 在新线程中启动服务器
    import threading
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(1)
    
    # 测试发送obs_dict
    async def run_test_client():
        test_dicts = [
            # 初始状态：只有手牌
            {
                'hand_cards': ['S2', 'H8'],
                'public_cards': [],
                'history': [ [(0, 3, [0, 1, 2, 3, 4])] ],
                'legal_actions': [0, 1, 2, 3, 4],
                'stakes': (96, 98),
                'current_player': 1
            },
            {
                'hand_cards': ['S2', 'H8'],
                'public_cards': [],
                'history': [],
                'legal_actions': [0, 1, 2, 3, 4],
                'stakes': (100, 100),
                'current_player': 0
            },
            {
                'hand_cards': ['S8', 'H8'],
                'public_cards': [],
                'history': [],
                'legal_actions': [0, 1, 2, 3, 4],
                'stakes': (5, 5),
                'current_player': 0
            },
            # Flop状态：3张公共牌
            {
                'hand_cards': ['S9', 'H8'],
                'public_cards': ['C8', 'D8', 'C5'],
                'history': [[(0, 1, [0, 1, 2]), (1, 1, [0, 1, 2])]],
                'legal_actions': [0, 1, 2, 3, 4],
                'stakes': (100, 100),
                'current_player': 0
            },
            # Turn状态：4张公共牌
            {
                'hand_cards': ['SA', 'CA'],
                'public_cards': ['DJ', 'CT', 'D6', 'CJ'],
                'history': [
                    [(0, 2, [0, 1, 2]), (1, 1, [0, 1, 2])],
                    [(1, 2, [0, 1, 2]), (0, 1, [0, 1, 2])]
                ],
                'legal_actions': [0, 1, 2, 3, 4],
                'stakes': (5, 5),
                'current_player': 1
            },

            {
                'hand_cards': ['H4', 'HA'],
                'public_cards': ['HJ', 'HQ', 'H9', 'S3'],
                'history': [
                    [(1, 3, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), (0, 1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])], 
                    [(0, 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), (1, 1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])], 
                    [(0, 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])], []
                ],
                'legal_actions': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'stakes': (49, 52),
                'current_player': 1
            },
            # River状态：5张公共牌
            {
                'hand_cards': ['SK', 'CK'],
                'public_cards': ['DJ', 'CT', 'D6', 'CJ', 'C9'],
                'history': [
                    [(0, 2, [0, 1, 2]), (1, 1, [0, 1, 2])],
                    [(0, 2, [0, 1, 2]), (1, 1, [0, 1, 2])],
                    [(0, 2, [0, 1, 2]), (1, 1, [0, 1, 2])]
                ],
                'legal_actions': [0, 1, 2, 3, 4],
                'stakes': (100, 100),
                'current_player': 1
            }
            
        ]
        
        for test_dict in test_dicts:
            print(f"\nSending: {test_dict}")
            await test_client(test_dict)
            await asyncio.sleep(1)  # 等待1秒再发送下一个
    
    # 运行测试客户端
    asyncio.run(run_test_client())


#test_obs_creation()
test_server()

if __name__ == "__main__":
    # 如果直接运行此文件，启动服务器
    start_server()
