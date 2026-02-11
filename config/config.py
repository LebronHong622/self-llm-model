# config/__init__.py
"""
多环境配置管理模块（增强版）

新增功能:
    - 配置热更新: 配置文件修改后自动重新加载
    - 配置验证: 参数合法性检查
    - 命令行参数覆盖

使用方法:
    from config import get_config, Environment
    
    # 基本使用
    config = get_config()
    
    # 启用热更新
    config = get_config(hot_reload=True)
    
    # 带验证的配置
    config = get_config(validate=True)
"""

import os
import yaml
import time
import threading
from typing import Literal, Optional, Dict, Any, List, Callable, get_type_hints
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# 配置目录
CONFIG_DIR = Path(__file__).parent


class Environment(Enum):
    """环境类型"""
    DEV = "dev"
    TEST = "test"
    PROD = "prod"
    
    @classmethod
    def from_string(cls, value: str) -> "Environment":
        try:
            return cls(value.lower())
        except ValueError:
            return cls.DEV


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


# ==================== 配置验证器 ====================

class ConfigValidator:
    """
    配置验证器（已弃用）

    注意：Pydantic 已经通过 Field 和 @model_validator 提供了完整的验证功能。
    此类保留用于向后兼容。
    """

    # 验证规则（仅供参考，实际验证由 Pydantic 处理）
    RULES = {
        "model.vocab_size": {"type": int, "min": 1000, "max": 1000000},
        "model.hidden_size": {"type": int, "min": 64, "max": 8192},
        "model.num_layers": {"type": int, "min": 1, "max": 100},
        "model.num_heads": {"type": int, "min": 1, "max": 128},
        "model.max_length": {"type": int, "min": 16, "max": 32768},
        "model.dropout": {"type": float, "min": 0.0, "max": 1.0},

        "training.batch_size": {"type": int, "min": 1, "max": 1024},
        "training.learning_rate": {"type": float, "min": 1e-8, "max": 1.0},
        "training.num_epochs": {"type": int, "min": 1, "max": 10000},
        "training.num_workers": {"type": int, "min": 0, "max": 64},
        "training.device": {"type": str, "choices": ["cuda", "cpu"]},

        "data.max_seq_length": {"type": int, "min": 16, "max": 32768},
    }

    @classmethod
    def validate(cls, config: BaseModel) -> List[str]:
        """
        验证配置（Pydantic 已自动处理）

        此方法仅返回空列表用于向后兼容。
        实际验证在 Config 初始化时自动完成。
        """
        # Pydantic 会在创建 Config 实例时自动验证
        # 如果需要手动触发验证，使用 config.model_validate(config.model_dump())
        return []

    @classmethod
    def _validate_custom(cls, config: "Config") -> List[str]:
        """自定义验证规则（已废弃，使用 Pydantic @model_validator）"""
        return []

    @staticmethod
    def _get_nested_value(data: Dict, path: str) -> Any:
        """获取嵌套字典值"""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value


# ==================== 配置热更新 ====================

class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监听器"""
    
    def __init__(self, config_manager: "ConfigManager"):
        self.config_manager = config_manager
        self.last_modified = 0
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # 防抖处理
        current_time = time.time()
        if current_time - self.last_modified < 1.0:
            return
        self.last_modified = current_time
        
        # 检查是否是配置文件
        if event.src_path.endswith(('.yaml', '.yml')):
            print(f"\n[ConfigWatcher] 检测到配置文件变更: {event.src_path}")
            self.config_manager.reload()
            print(f"[ConfigWatcher] 配置已重新加载\n")


class ConfigManager:
    """配置管理器（支持热更新）"""
    
    def __init__(self):
        self._config: Optional["Config"] = None
        self._config_paths: List[str] = []
        self._observer: Optional[Observer] = None
        self._callbacks: List[Callable] = []
        self._lock = threading.RLock()
    
    def load(
        self,
        env: Optional[Environment] = None,
        config_path: Optional[str] = None,
        validate: bool = False
    ) -> "Config":
        """
        加载配置
        
        Args:
            env: 环境类型
            config_path: 配置文件路径
            validate: 是否验证配置
        """
        with self._lock:
            # 确定加载路径
            if config_path:
                self._config = Config.from_yaml(config_path)
                self._config_paths = [config_path]
            elif env:
                self._config = Config.for_environment(env)
                self._config_paths = [
                    str(CONFIG_DIR / "base.yaml"),
                    str(CONFIG_DIR / f"{env.value}.yaml")
                ]
            else:
                env_str = os.getenv("SELF_LLM_ENV", "dev")
                environment = Environment.from_string(env_str)
                self._config = Config.for_environment(environment)
                self._config_paths = [
                    str(CONFIG_DIR / "base.yaml"),
                    str(CONFIG_DIR / f"{environment.value}.yaml")
                ]
            
            # 验证配置
            if validate:
                errors = ConfigValidator.validate(self._config)
                if errors:
                    error_msg = "\n".join([f"  - {e}" for e in errors])
                    raise ConfigValidationError(f"配置验证失败:\n{error_msg}")
            
            return self._config
    
    def reload(self) -> "Config":
        """重新加载配置"""
        with self._lock:
            if not self._config_paths:
                raise ValueError("没有可重载的配置路径")
            
            # 重新加载
            if len(self._config_paths) == 1:
                self._config = Config.from_yaml(self._config_paths[0])
            else:
                self._config = Config.from_multiple(
                    self._config_paths[0],
                    self._config_paths[1] if len(self._config_paths) > 1 else None
                )
            
            # 验证新配置
            try:
                # 触发 Pydantic 的验证
                self._config.model_validate(self._config.model_dump())
            except Exception as e:
                print(f"[ConfigManager] 配置验证失败: {e}")
                # 验证失败时可以选择回滚或抛出异常
                # 这里选择抛出异常让用户知道配置有问题
                raise

            # 执行回调
            for callback in self._callbacks:
                try:
                    callback(self._config)
                except Exception as e:
                    print(f"[ConfigManager] 回调执行失败: {e}")
            
            return self._config
    
    def start_watching(self):
        """启动文件监听（热更新）"""
        if self._observer:
            return

        self._observer = Observer()

        # 创建配置文件监听器
        watcher = ConfigFileWatcher(self)

        # 监听配置目录
        self._observer.schedule(watcher, str(CONFIG_DIR), recursive=False)
        self._observer.start()

        print(f"[ConfigManager] 已启动配置热更新监听: {CONFIG_DIR}")
    
    def stop_watching(self):
        """停止文件监听"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            print("[ConfigManager] 已停止配置热更新监听")
    
    def add_callback(self, callback: Callable[["Config"], None]):
        """添加配置变更回调"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[["Config"], None]):
        """移除配置变更回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    @property
    def config(self) -> Optional["Config"]:
        return self._config


# ==================== 配置数据类 ====================

class ModelConfig(BaseModel):
    """模型架构配置"""
    vocab_size: int = Field(default=10000, ge=1000, le=1000000, description="词表大小")
    input_size: int = Field(default=768, ge=64, le=8192, description="输入层大小")
    hidden_size: int = Field(default=768, ge=64, le=8192, description="隐藏层大小")
    context_length: int = Field(default=1024, ge=16, le=32768, description="上下文长度")
    num_layers: int = Field(default=12, ge=1, le=100, description="层数")
    num_heads: int = Field(default=12, ge=1, le=128, description="注意力头数")
    qwk_bias: bool = Field(default=False, description="是否使用偏置")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout率")
    norm_eps: float = Field(default=1e-5, ge=0.0, le=1.0, description="归一化epsilon")
    expansion_factor: int = Field(default=4, ge=1, le=100, description="前馈神经网络扩展因子")

class TrainingConfig(BaseModel):
    """训练配置"""
    batch_size: int = Field(default=32, ge=1, le=1024, description="批次大小")
    learning_rate: float = Field(default=1e-4, ge=1e-8, le=1.0, description="学习率")
    num_epochs: int = Field(default=10, ge=1, le=10000, description="训练轮数")
    num_workers: int = Field(default=4, ge=0, le=64, description="数据加载工作线程")
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="训练设备")
    save_interval: int = Field(default=1000, ge=1, description="保存间隔步数")
    log_interval: int = Field(default=100, ge=1, description="日志间隔步数")


class DataConfig(BaseModel):
    """数据配置"""
    dataset_path: str = Field(default="dataset/pretrain_hq.jsonl", description="数据集路径")
    tokenizer_path: str = Field(default="tokenizer", description="分词器路径")
    max_seq_length: int = Field(default=512, ge=16, le=32768, description="最大序列长度")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )


class Config(BaseModel):
    """总配置类"""
    model: ModelConfig = Field(default_factory=ModelConfig, description="模型配置")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="训练配置")
    data: DataConfig = Field(default_factory=DataConfig, description="数据配置")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="日志配置")
    environment: str = Field(default="dev", description="环境名称")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """验证环境名称"""
        valid_envs = {"dev", "test", "prod"}
        if v.lower() not in valid_envs:
            raise ValueError(f"environment 必须是 {valid_envs} 之一")
        return v.lower()

    @model_validator(mode="after")
    def validate_custom_rules(self) -> "Config":
        """自定义验证规则"""
        model = self.model

        # hidden_size 必须能被 num_heads 整除
        if model.hidden_size % model.num_heads != 0:
            raise ValueError(
                f"model.hidden_size ({model.hidden_size}) 必须能被 "
                f"model.num_heads ({model.num_heads}) 整除"
            )

        # max_seq_length 不能超过 context_length
        if self.data.max_seq_length > model.context_length:
            raise ValueError(
                f"data.max_seq_length ({self.data.max_seq_length}) 不能大于 "
                f"model.context_length ({model.context_length})"
            )

        # 检查数据集文件是否存在
        if not os.path.exists(self.data.dataset_path):
            raise ValueError(f"data.dataset_path: 文件不存在 '{self.data.dataset_path}'")

        # 检查分词器目录是否存在
        if not os.path.exists(self.data.tokenizer_path):
            raise ValueError(f"data.tokenizer_path: 目录不存在 '{self.data.tokenizer_path}'")

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """从单个 YAML 文件加载配置"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def from_multiple(cls, base_path: str, override_path: Optional[str] = None) -> "Config":
        """从多个 YAML 文件加载配置"""
        config = cls.from_yaml(base_path)

        if override_path and os.path.exists(override_path):
            with open(override_path, "r", encoding="utf-8") as f:
                override_data = yaml.safe_load(f) or {}

            base_dict = config.model_dump()
            merged_dict = cls._deep_merge(base_dict, override_data)
            config = cls._from_dict(merged_dict)

        return config

    @classmethod
    def for_environment(cls, env: Environment) -> "Config":
        """为指定环境加载配置"""
        base_path = CONFIG_DIR / "base.yaml"
        env_path = CONFIG_DIR / f"{env.value}.yaml"

        config = cls.from_multiple(
            str(base_path),
            str(env_path) if env_path.exists() else None
        )
        config.environment = env.value

        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """从字典创建配置对象"""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            environment=data.get("environment", "dev")
        )

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """深度合并两个字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    def to_yaml(self, yaml_path: str) -> None:
        """保存配置到 YAML 文件"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)


# ==================== 全局管理器 ====================

_manager = ConfigManager()


def get_config(
    env: Optional[Environment] = None,
    config_path: Optional[str] = None,
    hot_reload: bool = False,
    validate: bool = False,
    reload: bool = False
) -> Config:
    """
    获取配置实例
    
    Args:
        env: 指定环境类型
        config_path: 指定配置文件路径
        hot_reload: 是否启用热更新
        validate: 是否验证配置
        reload: 是否强制重新加载
    
    Returns:
        Config 实例
    """
    global _manager
    
    if reload or _manager.config is None:
        _manager.load(env, config_path, validate)
        
        if hot_reload:
            _manager.start_watching()
    
    return _manager.config


def reload_config() -> Config:
    """手动重新加载配置"""
    return _manager.reload()


def on_config_change(callback: Callable[[Config], None]):
    """注册配置变更回调"""
    _manager.add_callback(callback)


def stop_config_watcher():
    """停止配置监听"""
    _manager.stop_watching()


def set_environment(env: Environment) -> None:
    """设置当前环境"""
    os.environ["SELF_LLM_ENV"] = env.value
    _manager.load(env)


def validate_config(config: Optional[BaseModel] = None) -> List[str]:
    """
    验证配置

    Returns:
        错误信息列表，空列表表示验证通过
    """
    cfg = config or _manager.config
    if cfg is None:
        raise ValueError("没有加载的配置")

    # 使用 Pydantic 的验证
    try:
        cfg.model_validate(cfg.model_dump())
        return []
    except Exception as e:
        return [str(e)]


# 便捷函数
def get_dev_config(hot_reload: bool = False, validate: bool = False) -> Config:
    return get_config(Environment.DEV, hot_reload=hot_reload, validate=validate)


def get_test_config(hot_reload: bool = False, validate: bool = False) -> Config:
    return get_config(Environment.TEST, hot_reload=hot_reload, validate=validate)


def get_prod_config(hot_reload: bool = False, validate: bool = False) -> Config:
    return get_config(Environment.PROD, hot_reload=hot_reload, validate=validate)


# 打印配置
def print_config(config: Optional[Config] = None) -> None:
    cfg = config or get_config()
    errors = ConfigValidator.validate(cfg)
    
    print(f"\n{'='*60}")
    print(f"配置信息 [环境: {cfg.environment.upper()}]")
    if errors:
        print(f"⚠️  验证警告:")
        for e in errors:
            print(f"   - {e}")
    print(f"{'='*60}")
    print(f"\n模型配置:")
    print(f"  词表大小: {cfg.model.vocab_size}")
    print(f"  隐藏层: {cfg.model.hidden_size}")
    print(f"  层数: {cfg.model.num_layers}")
    print(f"  注意力头: {cfg.model.num_heads}")
    print(f"  最大长度: {cfg.model.max_length}")
    print(f"  Dropout: {cfg.model.dropout}")
    print(f"\n训练配置:")
    print(f"  批次大小: {cfg.training.batch_size}")
    print(f"  学习率: {cfg.training.learning_rate}")
    print(f"  设备: {cfg.training.device}")
    print(f"  轮数: {cfg.training.num_epochs}")
    print(f"  工作线程: {cfg.training.num_workers}")
    print(f"\n数据配置:")
    print(f"  数据集: {cfg.data.dataset_path}")
    print(f"  分词器: {cfg.data.tokenizer_path}")
    print(f"  最大序列长度: {cfg.data.max_seq_length}")
    print(f"\n日志配置:")
    print(f"  级别: {cfg.logging.level}")
    print(f"{'='*60}\n")
