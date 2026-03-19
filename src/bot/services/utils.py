import logging
import os

import streamlit as st
import yaml  # type: ignore[import]
from omegaconf import DictConfig, OmegaConf

# Example articles for few-shot learning
EXAMPLES_CLS_TOPIC = [
    {"text": "ШОК! Ученые из США открыли новый вид динозавров", "category": "Наука"},
    {"text": "Выборы президента страны пройдут в следующем месяце", "category": "Политика"},
    {"text": "Футбольный клуб нашего города выиграл чемпионат! Поздравляем!!!", "category": "Спорт"},
]


def load_config() -> DictConfig:
    """Load application configuration from YAML file."""
    with open("src/config/config.yaml", "r", encoding="utf-8") as file:
        return OmegaConf.create(yaml.safe_load(file))


def get_db_config() -> dict[str, str | None]:
    try:
        if hasattr(st, "secrets"):
            return {
                "user": st.secrets["POSTGRES_USER"],
                "password": st.secrets["POSTGRES_PASS"],
                "host": st.secrets["POSTGRES_HOST"],
                "port": st.secrets["POSTGRES_PORT"],
                "database": st.secrets["POSTGRES_DB"],
            }
    except Exception:
        logging.warning("Operation failed: using .env")

    return {
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASS"),
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "database": os.getenv("POSTGRES_DB"),
    }
