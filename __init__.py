# from envs.market import Market
# from envs.exchange import Exchange

from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
print(sys.path)