import yaml
from omegaconf import DictConfig, OmegaConf

# Example articles for few-shot learning
EXAMPLES_CLS_TOPIC = [
    {"text": "Ученые открыли новый вид динозавров", "category": "Наука"},
    {"text": "Выборы президента пройдут в следующем месяце", "category": "Политика"},
    {"text": "Футбольный клуб нашего города выиграл чемпионат!", "category": "Спорт"},
]


def load_config() -> DictConfig:
    """Load application configuration from YAML file."""
    with open("src/config/config.yaml", "r", encoding="utf-8") as file:
        return OmegaConf.create(yaml.safe_load(file))
