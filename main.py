from typing import Tuple
from dataclasses import dataclass

from utils import Config

import hydra
from hydra.core.config_store import ConfigStore


@hydra.main(version_base="1.2", config_path="configs/", config_name="config")
def main(config: Config) -> None:
    if config.algo.name == "me":
        import main_me as main
    #elif config.algo.name == "me_es":
    #    import main_me_es as main
    elif config.algo.name == "pga_me":
        import main_pga_me as main
    #elif config.algo.name == "qd_pg":
    #    import main_qd_pg as main
    elif config.algo.name == "dcg_me":
        import main_dcg_me as main
    elif config.algo.name == "mcpg_me":
        import main_mcpg_me as main
    elif config.algo.name == "memes":
        import main_memes as main
    elif config.algo.name == "mcpg_me_no_normalizer":
        import main_mcpg_me_no_normalizer as main
    elif config.algo.name == "mcpg_me_no_baseline":
        import main_mcpg_me_no_baseline as main
    elif config.algo.name == "mcpg_me_no_ppo_loss":
        import main_mcpg_me_no_ppo_loss as main
    
            
    else:
        raise NotImplementedError

    main.main(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()