import os
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from ruamel.yaml import YAML


LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ConfigModel(BaseModel):
    poe_api_key: str = Field(default="YOUR_API_KEY", description="POE API key")
    log_dir: str = Field(default="logs", description="Log directory")
    log_level: LOG_LEVEL = Field(default="INFO", description="Log level")

    host: str = Field(default="0.0.0.0", description="Host")
    port: int = Field(default=8800, description="Port")

    auth_secret: str | None = Field(default=None, description="Auth secret")


def load_config():
    config_path = os.environ.get("POE_API_CONFIG_PATH", "config.yaml")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
    with open(config_path, mode="r", encoding="utf-8") as f:
        yaml = YAML()
        config_dict = yaml.load(f) or {}
        model = ConfigModel.model_validate(config_dict)
        return model


def dump_config(model: ConfigModel, output_path):
    with open(output_path, mode="w", encoding="utf-8") as f:
        yaml = YAML()
        yaml.dump(model.model_dump(mode="json"), f)


CONFIG = load_config()


if __name__ == "__main__":
    config = ConfigModel()
    dump_config(config, "config.yaml.example")
    print("Example config file written.")
