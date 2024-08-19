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
        import main_me_mcpg as main
    
    elif config.algo.name == "ppoish_me":
        if config.algo.NORMALIZE_ENV:
            import main_ppoish_me_obs_norm as main
        else:
            if config.algo.sample_trajectory:
                import main_ppoish_me as main
            else:
                import main_ppoish_me_trans as main
            
    else:
        raise NotImplementedError

    main.main(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
