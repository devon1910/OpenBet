from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://openbet:openbet@localhost:5433/openbet"
    database_url_sync: str = "postgresql://openbet:openbet@localhost:5433/openbet"

    # Redis
    redis_url: str = "redis://localhost:6380/0"

    # API Keys
    football_data_api_key: str = ""
    api_football_key: str = ""
    anthropic_api_key: str = ""
    odds_api_key: str = ""

    # App
    log_level: str = "INFO"
    model_version: str = "v1"

    # Elo settings
    elo_k_factor: float = 32.0
    elo_k_factor_high_stakes: float = 40.0
    elo_home_advantage: float = 65.0
    elo_initial_rating: float = 1500.0

    # Betting thresholds
    straight_win_threshold: float = 0.60
    double_chance_threshold: float = 0.75
    max_picks_per_matchday: int = 12
    min_picks_per_matchday: int = 10

    # Claude reasoning
    claude_max_adjustment: float = 0.10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
