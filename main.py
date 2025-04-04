from typing import Tuple
from dataclasses import dataclass

from utils import Config

import hydra
from hydra.core.config_store import ConfigStore


@hydra.main(version_base="1.2", config_path="configs/", config_name="config")
def main(config: Config) -> None:
    if config.algo.name == "me":
        import main_me as main
    elif config.algo.name == "pga_me":
        import main_pga_me as main
    elif config.algo.name == "dcrl_me":
        import main_dcrl_me as main
    elif config.algo.name == "ascii_me":
        import main_ascii_me as main
    elif config.algo.name == "memes":
        import main_memes as main
    elif config.algo.name == "ppga":
        import main_ppga as main
            
    else:
        raise NotImplementedError

    main.main(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()